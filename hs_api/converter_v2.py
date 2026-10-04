from hs_api.neuron_models import IF_neuron, ANN_neuron, LIF_neuron
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any
import torch.nn.functional as F


class ModularSNNConverter:
    def __init__(self, energy_efficient_mode: bool = False, neuron_type: str = "ANN"):
        # Dispatch table mapping PyTorch types to conversion handlers
        self.registry: Dict[type, callable] = {}      #values of dictionary are functions that handle the conversion of each layer type
        self._register_default_handlers()
        self.energy_efficient_mode = energy_efficient_mode
        self.neuron_type = neuron_type
        self.output_layer_name = None  

    def register_handler(self, layer_type: type, handler_fn: callable):
        """Allows end-users to register custom layer converters (e.g., custom MaxPool, GAP)."""
        self.registry[layer_type] = handler_fn

    def _register_default_handlers(self):
        self.register_handler(nn.Conv2d, self._convert_conv2d)
        self.register_handler(nn.Linear, self._convert_linear)
        self.register_handler(nn.AdaptiveAvgPool2d, self._convert_gap)
        self.register_handler(nn.MaxPool2d, self._convert_maxpool)

    #convert fp32 weight and biases in model into int16
    @staticmethod
    def fp32_to_int16_state_dict(weights: dict):
        """Return two dicts:
        1. int16 weights   
        2. per‑tensor scale factors (float32)

        Biases are scaled using their corresponding weight scale
        """
        int16_sd, scales = {}, {}

        #First pass: compute and store scales for all weight tensors
        for name, tensor in weights.items():
            if name.endswith(".weight"):
                max_val = tensor.abs().max()
                if max_val == 0:
                    max_val = 1 #avoid divide-by-zero
                scale   = (2**15 - 1) / max_val
                int16_sd[name] = torch.round(tensor * scale).to(torch.int16)
                scales[name]   = scale.item()

        #Second pass: convert biases using the scale from corresponding weights
        for name, tensor in weights.items():
            if name.endswith(".bias"):
                weight_name = name.replace(".bias", ".weight")
                if weight_name not in scales:
                    raise ValueError(f"Missing corresponding weight tensor for bias: {name}")
                scale = scales[weight_name]
                int16_sd[name] = torch.round(tensor * scale).to(torch.int16)
                scales[name] = scale  # Store bias scale under its own name
        return int16_sd, scales

    def _trace_layer_shapes(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Tuple]:
        """Collects dynamic shape metadata for every layer using forward hooks.
        
        A forward hook is a custom function that is executed during the forward pass of a model. 
        It allows us to capture the input and output shapes of each layer without modifying the model's architecture or behavior.
        
        """
        shapes = {}
        def get_hook(name):
            def hook(module, input_tensor, output_tensor):
                shapes[name] = {
                    'in_shape': input_tensor[0].shape,
                    'out_shape': output_tensor.shape
                }
            return hook

        hooks = []
        for name, module in model.named_modules():
            # Skip the root container ("") and container modules that have sub-children
            if not name or len(list(module.children())) > 0:
                continue

            hooks.append(module.register_forward_hook(get_hook(name)))

        with torch.no_grad():
            model(sample_input)

        for h in hooks:
            h.remove()     #remove hooks after tracing to avoid side effects in future forward passes
        return shapes

    def _get_handler(self, module: nn.Module):
        """Finds handler for exact module type or subclass instance."""
        # 1. Exact match lookup (fast path)
        if type(module) in self.registry:
            return self.registry[type(module)]
        
        # 2. Inheritance check (handles QuantConv2d, QuantLinear etc.)
        for reg_cls, handler in self.registry.items():
            if isinstance(module, reg_cls):
                return handler
            
        return None
    def convert(self, model: nn.Module, weights_path: str, sample_input_shape=Tuple[int, ...]):
        """Primary endpoint for end-users."""
        # 1. Load full model structure and weights
        model = model
        weights = torch.load(weights_path, weights_only=True)  # Load the weights separately
        model.load_state_dict(weights)
        model.eval()
        neuron_type = self.neuron_type  # Use the neuron type specified during initialization

        # 2. Quantize weights & biases
        int16_sd, scales = self.fp32_to_int16_state_dict(weights)

        # 3. Trace shape execution graph
        dummy_input = torch.randn(sample_input_shape)
        shape_info = self._trace_layer_shapes(model, dummy_input)

        
        registered_layer_names = list(shape_info.keys())

        if registered_layer_names:
            print("Registered layer names in the model:")
            for name in registered_layer_names:
                print(f" - {name}")
            self.output_layer_name = registered_layer_names[-1]
            print(f"Output layer identified as: {self.output_layer_name}")
        else:
            raise ValueError("No registered layers found in the model. Please ensure that the model contains supported layer types.")

        # 4. Global IR representations
        axons = {}
        connections = {}
        outputs = []


        #5. State tracking
        prev_layer_name = None
        prev_layer_type = None
        prev_layer_idx = -1
        layer_counters = {"conv": 0, "linear": 0, "gap": 0}
        

        # 6. Dealing with energy-efficient mode (applicable to biases only)
        if not self.energy_efficient_mode:
            bias = ANN_neuron(theta=-1) # In non-energy-efficient mode, we create a bias neuron for each layer that has a bias term. This neuron will be connected to the feature map neurons in that layer.  

        else:
            bias = {}  # In energy-efficient mode, we skip bias neurons and use bias axons. Bias axons must be saved in a separate dictionary for later use in the CRI. 

        # 7. Modular conversion loop
        for name, module in model.named_modules():
            handler = self._get_handler(module)

            if handler is None:
                print(f"Skipping unregistered module type: {type(module).__name__} for layer {name}")
                continue  # Skip unregistered module types (e.g., Dropout, Flatten, etc.)

            #layer_shapes = shape_info[name]
            #neuron_type = self.get_neuron_type_string(module)

            # Delegate to layer handler
            prev_layer_type, prev_layer_idx, layer_counters = handler(
                module=module,
                layer_name=name,
                int16_sd=int16_sd,
                scales=scales,
                shapes=shape_info,
                axons=axons,
                connections=connections,
                outputs=outputs,
                prev_layer_name=prev_layer_name,
                prev_layer_type=prev_layer_type,
                prev_layer_idx=prev_layer_idx,
                layer_counters=layer_counters,
                neuron_type=neuron_type,
                bias = bias
            )

            prev_layer_name = name  # Update previous layer name for the next iteration

        if not self.energy_efficient_mode:
            print("Energy-efficient mode is OFF. Bias neurons have been created for layers with bias terms. Conversion returns only config dictionary for CRI.")
            return {'axons': axons, 'connections': connections, 'outputs': outputs} 

        else:
            print("Energy-efficient mode is ON. Bias axons have been created for layers with bias terms. Conversion returns config dictionary for CRI and bias axon dictionary. BOTH are used to run model on FPGA.")
            return {'axons': axons, 'connections': connections, 'outputs': outputs, 'bias_axons': bias}  # Return both the config dictionary and the bias axon dictionary

    @staticmethod
    def convert_bias_helper(connections: dict, axons: dict, layer_bias: torch.Tensor, feature_map: int, neuronName: str, bias_implementation: object):
        "bias_implementation can be either a dict (for energy-efficient mode) or an ANN neuron (for non-energy-efficient mode)"
        if isinstance(bias_implementation, dict):
            #implement bias using axon
            bias = layer_bias[feature_map-1]   #synaptic weight between bias axon and feature map neuron. feature_map-1 because feature_map starts from 1 while layer_bias is indexed from 0
            biasAxonName = "BiasA." + neuronName 
            axons[biasAxonName] = [(neuronName, bias)]    #add bias axon to axon dictionary
            layer_name = neuronName.split('.')[0]  # Extract layer name from neuronName (e.g., "C0" from "C0.1.1")
            if layer_name not in bias_implementation:
                bias_implementation[layer_name] = []
            bias_implementation[layer_name].append(biasAxonName)  # Store bias axon name
            return
        
        #implement bias using ANN neuron
        bias = layer_bias[feature_map-1]   #synaptic weight between bias neuron and feature map neruon
        biasNeuronName = "BiasN." + neuronName 
        connections[biasNeuronName] = ([(neuronName, bias)], bias_implementation)    #add bias neuron to connections dictionary
        
    # Handler implementations (accepting standardized parameters)
    def _convert_conv2d(self, **kwargs):
        # Conv2d specific logic using kwargs['shapes']['in_shape'], etc.
        # ...

        # 1. Extract only the variables THIS layer needs from kwargs
        int16_sd = kwargs['int16_sd']
        scales = kwargs['scales']
        layer_name = kwargs['layer_name']
        shapes = kwargs['shapes'][layer_name]               # From forward hook tracer
        connections = kwargs['connections']     # Global CRI connections dict
        axons = kwargs['axons']                   # Global CRI axons dict
        outputs = kwargs['outputs']               # Global CRI outputs list
        prev_layer_type = kwargs['prev_layer_type']
        prev_layer_idx = kwargs['prev_layer_idx']
        layer_counters = kwargs['layer_counters']
        module = kwargs['module']
        neuron_type = kwargs['neuron_type']
        bias = kwargs['bias']  # Bias neuron for this layer (if not in energy-efficient mode)

        # 2. Extract spatial metadata automatically from `module` and `shapes`
        _, in_c, in_h, in_w = shapes['in_shape']
        _, out_c, out_h, out_w = shapes['out_shape']
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding

        # 3. Retrieve weights, scale,bias, and neuron type for this specific layer name
        weight_key = f"{layer_name}.weight"
        bias_key = f"{layer_name}.bias"
        layer_weights = int16_sd[weight_key]
        layer_bias = int16_sd[bias_key] if bias_key in int16_sd else None

        if neuron_type == "IF":
            neuron = IF_neuron(theta=scales[weight_key])

        elif neuron_type == "LIF":
            neuron = LIF_neuron(theta=scales[weight_key])

        elif neuron_type == "ANN":
            neuron = ANN_neuron(theta=0)

        else:
            raise ValueError(f"Unsupported neuron type detected: {neuron_type} for layer {layer_name}")

        print(f"conv{prev_layer_idx+1} weight shape: {layer_weights.shape}")
        print(f"input shape: {shapes['in_shape']}, output shape: {shapes['out_shape']}")
        # 4. Check if prev_layer_type is None; if so, this is the first layer. Create axons and connections accordingly.
        if prev_layer_type is None:
            # Create input axons for the first layer
            for i in range(1, (in_c * in_h * in_w) + 1): # Axon indices start from 1
                axons[f"A{i}"] = []

            # axons -> C0 neurons
            axonMap = torch.arange(in_c * in_h * in_w, dtype=torch.float32).reshape(1, in_c, in_h, in_w)
            axonMap = axonMap + 1 #each entry labeled 1 to in_c * in_h * in_w to ignore zeros from padding          
            patchTensor = F.unfold(input=axonMap, kernel_size=kernel_size, stride=stride, padding=padding)   

            #patch_rows is a tensor where #rows = resolution of feature maps.
            #Each row contains the indices of the axons corresponding to each pixel in the feature map
            patch_rows = patchTensor.transpose(1, 2).squeeze(0) 
            patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor
            # for each axon index in patch_rows, create a connection to the corresponding C0 neurons with the appropriate weight from layer_weights
            for index , row in enumerate(patch_rows, start=1):
                for i, elem in enumerate(row):   #each elem in row is the axon index
                    axon_id = int(elem.item())
                    if axon_id != 0:     #avoid all zeros from padding 
                        key = f"A{axon_id}"     
                        #each axon index has a connection to one neuron in each feature map of the current layer. Iterate through each feature map and create an axonal synapse to the corresponding neuron in that feature map.
                        for feature_map, kernel in enumerate(layer_weights, start=1):
                            neuronName = f"C{prev_layer_idx+1}.{feature_map}.{index}"  #Create neuron entry C0.{feature map#}.{index}
                            if neuronName not in connections:  #first time this neuron is being added, initialize its connection entry
                                connections[neuronName] = ([], neuron) 
                                # first time this neuron is being added, implement bias if applicable
                                if layer_bias is not None:
                                    self.convert_bias_helper(connections, axons, layer_bias, feature_map, neuronName, bias) 
                                # for the first time this neuron is being added, check if it is the output layer and add to outputs list if so
                                if layer_name == self.output_layer_name:   #if this is the output layer, add neurons to outputs list
                                    outputs.append(neuronName)
                            flat_kernel = kernel.flatten()
                            weight = flat_kernel[i].item()
                            axons[key].append((neuronName, weight))
        
        # 3. Create new layer nodes for this Conv2d layer
        else:
            #creating cMap to identify which conv neuron from the previous layer is connected to which conv neuron in the current layer.
            cMap = torch.arange(in_h * in_w, dtype=torch.float32).reshape(1, 1, in_h, in_w)
            cMap = cMap + 1          
            patchTensor = F.unfold(input=cMap, kernel_size=kernel_size, stride=stride, padding=padding)   # dilation=1 by default

            #patch_rows is a tensor where #rows = resolution of feature maps.
            #Each row contains the indices of the axons corresponding to each pixel in the feature map
            patch_rows = patchTensor.transpose(1, 2).squeeze(0)
            patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor
                    
            #iterate through each patch row. #rows = resolution of output feature map 
            for j, row in enumerate(patch_rows, start=1):
                #iterate through each elem in row. Each elem is index of C1 -> C2 
                for i, elem in enumerate(row):
                    index = int(elem.item())
                    if index != 0:
                        for output_idx, output_channel in enumerate(layer_weights, start=1): 
                            neuronName = f"C{prev_layer_idx+1}.{output_idx}.{j}"  #each row corresponds to one pixel in feature map. Create neuron entry C{layer_index}.{feature map#}.{index}
                            if neuronName not in connections:  #first time this neuron is being added, initialize its connection entry
                                connections[neuronName] = ([], neuron) 
                                #this is first time this neuron is being added, implement bias if applicable
                                if layer_bias is not None:
                                    self.convert_bias_helper(connections, axons, layer_bias, output_idx, neuronName, bias) 
                                #this is first time this neuron is being added, check if it is the output layer and add to outputs list if so
                                if layer_name == self.output_layer_name:   
                                    outputs.append(neuronName)
                                    
                            #inner loop: iterate over input-channel kernels with index for this output channel
                            for feature_map, kernel in enumerate(output_channel, start=1):
                                key = f"C{prev_layer_idx}.{feature_map}.{index}"
                                flat_kernel = kernel.flatten() #flatten kernel is 1D tensor of the weights for C1 -> C2
                                weight = flat_kernel[i].item()
                                connections[key][0].append((neuronName, weight))

        prev_layer_type = "conv"
        prev_layer_idx += 1
        layer_counters['conv'] += 1
    
        return prev_layer_type, prev_layer_idx, layer_counters

    def _convert_linear(self, **kwargs):
        # Linear logic
        # ...
        # 1. Extract only the variables THIS layer needs from kwargs
        int16_sd = kwargs['int16_sd']
        scales = kwargs['scales']
        layer_name = kwargs['layer_name']
        shapes = kwargs['shapes'][layer_name]              # From forward hook tracer
        prev_layer_name = kwargs['prev_layer_name']
        connections = kwargs['connections']     # Global CRI connections dict
        axons = kwargs['axons']                   # Global CRI axons dict
        outputs = kwargs['outputs']               # Global CRI outputs list
        prev_layer_type = kwargs['prev_layer_type']
        prev_layer_idx = kwargs['prev_layer_idx']
        layer_counters = kwargs['layer_counters']
        module = kwargs['module']
        layer_name = kwargs['layer_name']
        neuron_type = kwargs['neuron_type']
        bias = kwargs['bias']  # Bias neuron for this layer (if not in energy-efficient mode)

        # 2. Extract spatial metadata automatically from `module` and `shapes`
        in_h, in_w = shapes['in_shape']
        out_h, out_w = shapes['out_shape']

        # 3. Retrieve weights, scale,bias, and neuron type for this specific layer name
        weight_key = f"{layer_name}.weight"
        bias_key = f"{layer_name}.bias"
        layer_weights = int16_sd[weight_key]
        layer_bias = int16_sd[bias_key] if bias_key in int16_sd else None

        if neuron_type == "IF":
            neuron = IF_neuron(theta=scales[weight_key])

        elif neuron_type == "LIF":
            neuron = LIF_neuron(theta=scales[weight_key])

        elif neuron_type == "ANN":
            neuron = ANN_neuron(theta=0)

        else:
            raise ValueError(f"Unsupported neuron type detected: {neuron_type} for layer {layer_name}")

        print(f"linear{prev_layer_idx+1} weight shape: {layer_weights.shape}")
        print(f"input shape: {shapes['in_shape']}, output shape: {shapes['out_shape']}")

        # 4. Check if prev_layer_type is None; if so, this is the first layer. Create axons and connections accordingly.
        if prev_layer_type is None and prev_layer_name is None:
            # Create input axons for the first layer
            for i in range(in_h * in_w):
                allConnections = []
                for j, weight in enumerate(layer_weights[:, i], start=1):
                    neuronName = f"FC{prev_layer_idx+1}.{j}"
                    if neuronName not in connections:  #first time this neuron is being added, initialize its connection entry
                        connections[neuronName] = ([], neuron)
                    connectingNeuron = (neuronName, weight.item())
                    allConnections.append(connectingNeuron)
                axons[f"A{i+1}"] = allConnections

            if layer_name == self.output_layer_name:   #if this is the last layer, add neurons to outputs list
                for neuron in allConnections:
                    outputs.append(neuron[0])  #add neuronName to outputs list
            
        # 5. Check if prev_layer_type is "conv", "linear", "GAP", or "MaxPool" to determine how to connect the previous layer's neurons to this linear layer's neurons
        elif prev_layer_type == "conv":
            # connect conv neurons to linear neurons
            _, prev_in_c, prev_in_h, prev_in_w = kwargs['shapes'][prev_layer_name]['out_shape']  #extract output shape of previous conv layer
            feature_map = 1
            for col in range(layer_weights.shape[1]):  #x.shape[1] == number of col
                allConnections = []
                if col % (prev_in_h * prev_in_w) == 0 and col != 0:  
                    feature_map += 1
                for i, elem in enumerate(layer_weights[:, col], start=1):     #iterate over element in a col
                    neuronName = f"FC{prev_layer_idx+1}.{i}"
                    if neuronName not in connections:
                        connections[neuronName] = ([], neuron)
                    connectingNeuron = (neuronName, elem.item())
                    allConnections.append(connectingNeuron)

                for connectingNeuron in allConnections:
                    connections[f"C{prev_layer_idx}.{feature_map}.{(col % (prev_in_h * prev_in_w)) + 1}"][0].append(connectingNeuron)

            if layer_name == self.output_layer_name:   #if this is the last layer, add neurons to outputs list
                for neuron in allConnections:
                    outputs.append(neuron[0])  #add neuronName to outputs list
            
        elif prev_layer_type == "linear":
            # connect linear neurons to linear neurons
            for col in range(layer_weights.shape[1]):  #x.shape[1] == number of col
                allConnections = []
                for i, elem in enumerate(layer_weights[:, col], start=1):     #iterate over element in a col
                    neuronName = f"FC{prev_layer_idx+1}.{i}"
                    if neuronName not in connections:
                        connections[neuronName] = ([], neuron)
                    connectingNeuron = (neuronName, elem.item())
                    allConnections.append(connectingNeuron)

                for connectingNeuron in allConnections:
                    connections[f"FC{prev_layer_idx}.{col+1}"][0].append(connectingNeuron)

            if layer_name == self.output_layer_name:   #if this is the last layer, add neurons to outputs list
                for neuron in allConnections:
                    outputs.append(neuron[0])  #add neuronName to outputs list

        elif prev_layer_type == "GAP":
            return 0  # Placeholder for GAP to Linear conversion logic

        elif prev_layer_type == "MaxPool":
            return 0  # Placeholder for MaxPool to Linear conversion logic
    
        prev_layer_type = "linear"
        prev_layer_idx += 1
        layer_counters['linear'] += 1
            
        return prev_layer_type, prev_layer_idx, layer_counters

    def _convert_gap(self, **kwargs):
        # Global Average Pooling logic
        # ...
        return 0

    def _convert_maxpool(self, **kwargs):
        # Max Pooling logic
        # ...
        return 0