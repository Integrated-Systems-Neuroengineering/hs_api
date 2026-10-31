"""
PyTorch-to-CRI network converter (version 2).

This module converts a trained PyTorch nn.Module into the network
representation read by the CRI neuromorphic hardware/simulator: a dictionary of
input axons, a dictionary of neurons with their outgoing synapses, and a list of
output neurons.

The conversion is *modular*: every supported PyTorch layer type is mapped to a
handler method through a dispatch table (ModularSNNConverter.registry).
Users can extend the converter with new layer types by calling
register_handler().

Supported layers (default handlers)
-----------------------------------
- nn.Conv2d            -> _convert_conv2d()
- nn.Linear            -> _convert_linear()
- nn.AdaptiveAvgPool2d -> _convert_gap()
  (global average pooling only, i.e. output size (1, 1))
- nn.MaxPool2d         -> _convert_maxpool()

Subclasses of these types (e.g. quantized QuantConv2d / QuantLinear
layers) are handled by the handler of their nearest registered base class.
Modules without a handler (ReLU, Flatten, Dropout, ...) are
skipped; their behaviour is assumed to be absorbed by the spiking neuron model
or by the implicit flattening order described below.

Output format
-------------
convert() returns a dictionary with the keys

axons : dict[str, list[tuple[str, int]]]
    Maps an axon name to its list of (target_neuron_name, weight) synapses.
connections : dict[str, tuple[list[tuple[str, int]], neuron_model]]
    Maps a neuron name to (synapse_list, neuron_model) where
    synapse_list holds (target_neuron_name, weight) tuples and
    neuron_model is an instance from hs_api.neuron_models.
outputs : list[str]
    Names of the neurons belonging to the final (output) layer.
bias_axons : dict[str, list[str]]
    Only present when energy_efficient_mode=True. Maps a CRI layer name
    (e.g. "conv1") to the bias axon names that feed that layer.

Naming conventions
------------------
All indices are 1-based.

- Input axons: A{i} where i enumerates the flattened input tensor in
  PyTorch (C, H, W) row-major order.
- Spatial neurons (conv / maxpool): {type}{layer}.{channel}.{pixel},
  e.g. conv1.3.17 is pixel 17 (row-major) of output channel 3 of the
  first converted layer.
- Non-spatial neurons (linear / gap): {type}{layer}.{neuron},
  e.g. linear4.10.
- Bias neurons (standard mode): BiasN.{k}.
- Bias axons (energy-efficient mode): BiasA.{group}.{k}, where group
  is the target neuron name with its last index removed.

{layer} is a single global counter shared by all layer types, so a
LeNet-style network produces conv1, maxpool2, conv3, maxpool4,
linear5, ... rather than a separate counter per type.

Assumptions and limitations
---------------------------
- The network must be a purely feed-forward chain. Layers are converted in
  the order returned by model.named_modules() (i.e. definition order),
  which must match the execution order. Residual/skip connections and
  branching topologies are not supported.
- The sample input passed to convert() must have a
  batch size of 1.
- A Linear layer that follows a spatial layer assumes the activations were
  flattened in PyTorch's default channel-major (C, H, W) order.

Example
-------
>>> converter = ModularSNNConverter(neuron_type="IF",
...                                 quantization_aware_training=True)
>>> config = converter.convert(model, "lenet5.pth", (1, 1, 28, 28))
>>> config["axons"], config["connections"], config["outputs"]
"""

from hs_api.neuron_models import IF_neuron, ANN_neuron, LIF_neuron
import torch
import torch.nn as nn
from typing import Dict, Tuple, Any
import torch.nn.functional as F
import warnings

class ModularSNNConverter:
    """
    Convert a feed-forward PyTorch model into a CRI axon/neuron network.

    The converter loads weights into the model, quantizes them to int16 (or
    validates that they already are), traces the tensor shape at every layer
    with a sample input, and then walks the model layer by layer, delegating
    each layer to the handler registered for its type. Handlers append to
    shared axons, connections and outputs structures that together
    form the CRI network description.

    Parameters
    ----------
    energy_efficient_mode : bool, optional
        Controls how bias terms are implemented. If False (default),
        biases are implemented with always-firing bias *neurons*
        (BiasN.k) placed inside the network. If True, biases are
        implemented with bias *axons* (BiasA.*) that are driven from the
        host, which avoids spending neuron resources and spike traffic on
        bias neurons. In this mode the bias axon names are also returned
        under "bias_axons" so the host knows which axons to drive.
    quantization_aware_training : bool, optional
        If True, the supplied weights are assumed to be floating point
        (e.g. from a QAT model) and are quantized to int16 with a per-tensor
        symmetric scale by fp32_to_int16_state_dict(). If False
        (default), the weights are assumed to be quantized already; any
        non-int16 tensor is cast to int16 with a warning, and the per-tensor
        scales are requested interactively via get_scales().
    neuron_type : {"ANN", "IF", "LIF"}, optional
        Neuron model used for every converted layer. "ANN" (default)
        creates memory-less ANN_neuron
        instances with theta=0. "IF" / "LIF" create
        integrate-and-fire / leaky integrate-and-fire neurons whose threshold
        is derived from the layer's quantization scale (weighted layers) or
        from the pooling window (pooling layers).

    Attributes
    ----------
    MAX_AXON_NUMBER : int
        Maximum number of axons supported by the hardware.
    MAX_AXONAL_FANOUT : int
        Maximum number of synapses a single axon may have. Enforced when
        creating bias axons.
    MAX_AXONAL_FANIN : int
        Maximum number of axons that may target a single neuron.
    MAX_NEURONAL_FANOUT : int
        Maximum number of synapses a single neuron may have. Enforced when
        creating bias neurons.
    MAX_NEURONAL_FANIN : int
        Maximum number of neurons that may target a single neuron.
    registry : dict[type, callable]
        Dispatch table mapping nn.Module subclasses to handler callables.
    energy_efficient_mode : bool
        See the parameter of the same name.
    quantization_aware_training : bool
        See the parameter of the same name.
    neuron_type : str
        See the parameter of the same name.
    output_layer_name : str or None
        Fully-qualified PyTorch name of the module treated as the output
        layer. Set by convert(); None before conversion.

    Notes
    -----
    Only MAX_AXONAL_FANOUT and MAX_NEURONAL_FANOUT are currently
    enforced (for bias axons/neurons). The remaining limits are recorded for
    reference and are not checked by this class.
    """

    #hardware constraints
    MAX_AXON_NUMBER = 16352
    MAX_AXONAL_FANOUT = 4096
    MAX_AXONAL_FANIN = 8192
    MAX_NEURONAL_FANOUT = 4095
    MAX_NEURONAL_FANIN = 8159

    def __init__(self, energy_efficient_mode: bool = False, quantization_aware_training: bool = False, neuron_type: str = "ANN"):
        """
        Initialize the converter and register the default layer handlers.

        See the class docstring for a description of each parameter.
        """
        # Dispatch table mapping PyTorch types to conversion handlers
        self.registry: Dict[type, callable] = {}      #values of dictionary are functions that handle the conversion of each layer type
        self._register_default_handlers()
        self.energy_efficient_mode = energy_efficient_mode
        self.quantization_aware_training = quantization_aware_training
        self.neuron_type = neuron_type
        self.output_layer_name = None  

    def register_handler(self, layer_type: type, handler_fn: callable):
        """Allows end-users to register custom layer converters (e.g., custom MaxPool, GAP).

        Registering a handler for a type that is already registered replaces
        the existing handler.

        Parameters
        ----------
        layer_type : type
            nn.Module subclass the handler converts. Instances of
            subclasses of layer_type are also dispatched to this handler
            unless a more specific type is registered (see
            _get_handler()).
        handler_fn : callable
            Function called as handler_fn(**kwargs) for each matching
            layer. It receives the following keyword arguments:

            - module (nn.Module): the layer being converted.
            - layer_name (str): fully-qualified PyTorch module name.
            - int16_sd (dict): quantized state dict.
            - scales (dict): per-tensor quantization scales.
            - shapes (dict): output of _trace_layer_shapes() for
              the whole model, keyed by module name.
            - axons, connections, outputs: the shared CRI
              structures, to be mutated in place.
            - prev_layer_name (str or None), prev_layer_type
              (str or None), prev_layer_idx (int): state describing the
              previously converted layer.
            - layer_counters (dict[str, int]): per-type layer counts.
            - neuron_type (str): "ANN", "IF" or "LIF".
            - bias: bias implementation object; see
              convert_bias_helper().

            It must return the tuple
            (prev_layer_type, prev_layer_idx, layer_counters) describing
            the state *after* this layer, which is passed to the next handler.
            Handlers should accept **kwargs so that new arguments can be
            added without breaking them.

        Examples
        --------
        >>> def convert_avgpool(**kwargs):
        ...     ...
        ...     return "avgpool", kwargs["prev_layer_idx"] + 1, kwargs["layer_counters"]
        >>> converter.register_handler(nn.AvgPool2d, convert_avgpool)
        """
        self.registry[layer_type] = handler_fn

    def _register_default_handlers(self):
        """Register the built-in handlers for Conv2d, Linear, AdaptiveAvgPool2d and MaxPool2d."""
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

        Each *.weight tensor is quantized with a symmetric per-tensor
        scale that maps its largest absolute value to the int16 maximum:

            scale = (2**15 - 1) / max(|W|)
            W_int16 = round(W * scale)

        A *.bias tensor reuses the scale of the matching *.weight
        tensor so that W x + b is scaled uniformly and the layer output
        remains a constant multiple of the floating-point output.

        Parameters
        ----------
        weights : dict[str, torch.Tensor]
            Floating-point state dict. Every bias key must have a matching
            weight key ("<layer>.bias" -> "<layer>.weight").

        Returns
        -------
        int16_sd : dict[str, torch.Tensor]
            Quantized weight and bias tensors (dtype torch.int16). Keys
            that end in neither .weight nor .bias (e.g. BatchNorm
            running statistics) are dropped.
        scales : dict[str, float]
            Scale factor applied to each quantized tensor, keyed by tensor
            name.

        Raises
        ------
        ValueError
            If a bias tensor has no corresponding weight tensor.

        Notes
        -----
        An all-zero weight tensor is given max_val = 1 to avoid division
        by zero. Bias values whose scaled magnitude exceeds the int16 range
        are not clipped and will overflow when cast.
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

    @staticmethod
    def get_weights(weights):
        """
        Load a state dict from a file path, or pass an existing one through.

        Parameters
        ----------
        weights : str or dict
            Path to a .pth file containing a state dict, or a state dict.
            Files are loaded onto the CPU with weights_only=True.

        Returns
        -------
        dict[str, torch.Tensor]
            The state dict.

        Raises
        ------
        TypeError
            If weights is neither a str nor a dict.
        """
        if isinstance(weights, str):  #path to weights provided 
            weights = torch.load(weights, map_location="cpu", weights_only=True)

        elif not isinstance(weights, dict):
            raise TypeError("Expected a .pth file path or a state_dict")

        return weights

    @staticmethod
    def get_scales(weights):  #used if quantization utilized is not quantization aware training
        """
        Interactively ask the user for the quantization scale of each tensor.

        Used when the weights were quantized outside this converter, so the
        scale that maps them back to floating point is not known. The user is
        first asked whether a single scale applies to all tensors; if not,
        one scale is requested per tensor.

        Parameters
        ----------
        weights : dict[str, torch.Tensor]
            State dict whose keys determine which scales are requested.

        Returns
        -------
        dict[str, float]
            Scale for every key in weights.

        Raises
        ------
        ValueError
            If the yes/no answer is not "y" or "n", or if an entered
            scale cannot be parsed as a float.

        Notes
        -----
        This method blocks on input() and therefore cannot be used in
        non-interactive environments (batch jobs, CI, some notebooks).
        """
        scales = {}

        same_scale = input("Apply the same scale to all weights? (y/n): ").strip().lower()

        if same_scale == "y":
            scale = float(input("Enter the scale value for all weights: "))

            for key in weights.keys():
                scales[key] = scale

        elif same_scale == "n":
            for key in weights.keys():
                scale = float(input(f"Enter the scale value for '{key}': "))
                scales[key] = scale

        else:
            raise ValueError("Invalid input. Please enter 'y' or 'n'.")
        
        return scales
    
    def _trace_layer_shapes(self, model: nn.Module, sample_input: torch.Tensor) -> Dict[str, Tuple]:
        """Collects dynamic shape metadata for every layer using forward hooks.
        
        A forward hook is a custom function that is executed during the forward pass of a model. 
        It allows us to capture the input and output shapes of each layer without modifying the model's architecture or behavior.

        Hooks are attached to every *leaf* module (modules with no children),
        including modules that have no conversion handler such as ReLU or
        Flatten. One forward pass is run under torch.no_grad() and all
        hooks are removed afterwards.

        Parameters
        ----------
        model : nn.Module
            Model to trace. Should already be in eval() mode.
        sample_input : torch.Tensor
            Example input with batch size 1.

        Returns
        -------
        dict[str, dict[str, torch.Size]]
            Maps each leaf module name to
            {'in_shape': torch.Size, 'out_shape': torch.Size}. Shapes
            include the batch dimension. Entries are inserted in execution
            order, so the last key is the last leaf module that ran. Only
            the first positional input of each module is recorded.
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
        """Finds handler for exact module type or subclass instance.

        Parameters
        ----------
        module : nn.Module
            Module to look up.

        Returns
        -------
        callable or None
            Handler registered for type(module). If there is none, the
            first registered type (in registration order) that module is
            an instance of. None if no handler matches.
        """
        # 1. Exact match lookup (fast path)
        if type(module) in self.registry:
            return self.registry[type(module)]
        
        # 2. Inheritance check (handles QuantConv2d, QuantLinear etc.)
        for reg_cls, handler in self.registry.items():
            if isinstance(module, reg_cls):
                return handler
            
        return None
    def convert(self, model: nn.Module, weights, sample_input_shape: Tuple[int, ...]):
        """Primary endpoint for end-users.

        Runs the full conversion pipeline:

        1. Load weights into model and switch it to eval mode.
        2. Quantize the weights (QAT mode) or validate/cast them to int16 and
           request their scales interactively (non-QAT mode).
        3. Trace per-layer input/output shapes with a random input of shape
           sample_input_shape and identify the output layer as the last
           leaf module to execute.
        4. Create the first bias neuron BiasN.1 (standard mode) or an
           empty bias-axon dict (energy-efficient mode).
        5. Walk model.named_modules() and dispatch every module that has
           a registered handler, threading the previous-layer state through
           consecutive handlers.
        6. Standard mode only: remove BiasN.1 if no layer used it.

        Parameters
        ----------
        model : nn.Module
            Model architecture. Its parameter names must match the keys of
            weights. Layers must be defined in the same order in which
            they execute (see module docstring).
        weights : str or dict
            Path to a .pth state dict or the state dict itself. In
            non-QAT mode, non-int16 tensors in this dict are replaced in
            place with int16 copies.
        sample_input_shape : tuple of int
            Shape of one model input including a batch dimension of 1, e.g.
            (1, 1, 28, 28) for MNIST or (1, 784) for an MLP.

        Returns
        -------
        dict
            {'axons': ..., 'connections': ..., 'outputs': ...} and, in
            energy-efficient mode, 'bias_axons' as well. See the module
            docstring for the structure of each entry.

        Raises
        ------
        TypeError
            If weights is neither a path nor a dict.
        RuntimeError
            If weights does not match the model's parameters
            (raised by load_state_dict).
        ValueError
            If tracing finds no leaf modules, if neuron_type is not
            supported, or if a layer follows a layer type its handler cannot
            connect to.

        Notes
        -----
        The output layer is the last *leaf module* executed during tracing.
        If the model ends with a module that has no handler (e.g. a final
        ReLU or Softmax), no layer matches it and outputs will be
        empty.

        Progress information (layer names, shapes, mode) is printed to
        stdout.
        """
        # 1. Load full model structure and weights
        weights = self.get_weights(weights)  #weights variable can be path or dict
        model.load_state_dict(weights)
        model.eval()
        neuron_type = self.neuron_type  # Use the neuron type specified during initialization

        # 2. Quantize weights & biases

        #applies quantization to QAT weights
        if self.quantization_aware_training == True:
            int16_sd, scales = self.fp32_to_int16_state_dict(weights)

        #assumes weights are already quantized using some method. Checks that weights are int16
        else:
            for key, tensor in weights.items():
                if tensor.dtype != torch.int16:
                    old_dtype = tensor.dtype

                    weights[key] = tensor.to(torch.int16)

                    warnings.warn(
                        f"Weight '{key}' converted from {old_dtype} "
                        f"to {weights[key].dtype}. Values may lose precision.",
                        UserWarning
                    )

            scales = self.get_scales(weights)
            int16_sd = weights
            

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
        prev_layer_idx = 0   #layers use 1-based indexing for naming
        layer_counters = {"conv": 0, "linear": 0, "gap": 0, "maxpool": 0}  # Track the number of each layer type processed
        

        # 6. Dealing with energy-efficient mode (applicable to biases only)
        if not self.energy_efficient_mode:
            bias = ["BiasN.1", ANN_neuron(theta=-1)] # In non-energy-efficient mode, we create a bias neuron for each layer that has a bias term. This neuron will be connected to the feature map neurons in that layer.
            connections[bias[0]] = ([], bias[1])   #first bias neuron will be created regardless if biases are utilized in the model
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

        if not self.energy_efficient_mode and len(connections["BiasN.1"][0]) == 0:   # BiasN.1 only exists in non-energy-efficient mode
            connections.pop("BiasN.1")   #remove first bias neuron in connections if it has no outgoing synapses

        if not self.energy_efficient_mode:
            print("Energy-efficient mode is OFF. Bias neurons have been created for layers with bias terms. Conversion returns only config dictionary for CRI.")
            return {'axons': axons, 'connections': connections, 'outputs': outputs} 

        else:
            print("Energy-efficient mode is ON. Bias axons have been created for layers with bias terms. Conversion returns config dictionary for CRI and bias axon dictionary. BOTH are used to run model on FPGA.")
            return {'axons': axons, 'connections': connections, 'outputs': outputs, 'bias_axons': bias}  # Return both the config dictionary and the bias axon dictionary

    def convert_bias_helper(self, connections: dict, axons: dict, layer_bias: torch.Tensor, feature_map: int, neuronName: str, bias_implementation: object, bias_tracker: dict):
        """
        Bias_implementation can be either a dict (for energy-efficient mode) or a list containing two elements: the name of the current bias neuron and the ANN neuron object (for non-energy-efficient mode)

        Add the bias synapse for one neuron.

        Called once per neuron, the first time that neuron is created by a
        weighted layer handler. A bias is modelled as a synapse from a
        bias neuron (constantly active) or bias axon (actived at specific timestep)
        to the neuron, with the bias value as the synaptic weight.

        Standard mode (bias_implementation is a list)
            The synapse is added to the current bias neuron
            bias_implementation[0]. A single bias neuron serves every
            layer until it reaches MAX_NEURONAL_FANOUT synapses, after
            which a new bias neuron BiasN.{k+1} is created and
            bias_implementation[0] is updated in place so later calls use
            it.

        Energy-efficient mode (bias_implementation is a dict)
            The synapse is added to a bias axon named
            BiasA.{group}.{k}, where group is neuronName without
            its last index (one group per output channel for conv layers,
            one group per layer for linear layers). When an axon reaches
            MAX_AXONAL_FANOUT synapses a new axon with index k+1 is
            created. Each layer has its own corresponding set of bias axons. 
            No bias axon has synapses to neurons belonging to different layers.
            The axon name is recorded in bias_implementation[layer] 
            so the user knows which axons to activate for a given layer.

        Parameters
        ----------
        connections : dict
            Global CRI connections dict (mutated in standard mode).
        axons : dict
            Global CRI axons dict (mutated in energy-efficient mode).
        layer_bias : torch.Tensor
            Quantized bias vector of the current layer, one value per output
            channel / output neuron.
        feature_map : int
            1-based output channel (conv) or output neuron (linear) index;
            selects layer_bias[feature_map - 1].
        neuronName : str
            Name of the neuron receiving the bias.
        bias_implementation : list or dict
            [current_bias_neuron_name, bias_neuron_model] in standard
            mode, or the bias-axon dict in energy-efficient mode. Mutated in
            place.
        bias_tracker : dict[str, int]
            Per-layer state mapping a bias group to its current bias axon
            index. Only used in energy-efficient mode.

        Returns
        -------
        None

        Notes
        -----
        In energy-efficient mode the axon name is appended to
        bias_implementation[layer] once per target neuron, so the list
        may contain repeated names.
        """

        if isinstance(bias_implementation, dict):
            #implement bias using axon
            layer_name = neuronName.split('.')[0]  # Extract layer name from neuronName (e.g., "conv1" from "conv1.1.1")
            if layer_name not in bias_tracker:
                bias_tracker[layer_name] = 1  #first bias neuron or axon has index 1 for the given output channel
            biasIndex = bias_tracker[layer_name]
            biasAxonName = f"BiasA.{layer_name}.{biasIndex}"   
            if biasAxonName not in axons:
                axons[biasAxonName] = []  #create bias axon in axon dictionary
            else:
                if len(axons[biasAxonName]) == self.MAX_AXONAL_FANOUT:   #check if axon fan-out equals limit
                    print(f"{biasAxonName} exceeds max axonal fanout. Creating another bias axon." )
                    biasAxonName = f"BiasA.{layer_name}.{biasIndex+1}"
                    bias_tracker[layer_name] += 1
                    axons[biasAxonName] = []  #create bias axon in axon dictionary

            bias = layer_bias[feature_map-1]   #synaptic weight between bias axon and feature map neuron. feature_map-1 because feature_map starts from 1 while layer_bias is indexed from 0
            axons[biasAxonName].append((neuronName, bias))   #add bias axon to axon dictionary
            
            if layer_name not in bias_implementation:
                bias_implementation[layer_name] = []
            bias_implementation[layer_name].append(biasAxonName)  # Store bias axon name
            return
        
        #implement bias using ANN neuron
        biasNeuronName = bias_implementation[0]  #name of current bias neuron
        biasNeuronModel = bias_implementation[1]
        bias = layer_bias[feature_map-1]   #synaptic weight between bias neuron and feature map neuron
        biasIndex = int(biasNeuronName.split(".")[1])
        if len(connections[biasNeuronName][0]) == self.MAX_NEURONAL_FANOUT:
            print(f"{biasNeuronName} exceeds max neuronal fan out. Creating another bias neuron.")
            biasNeuronName = f"BiasN.{biasIndex+1}"    #change name of current bias neuron
            connections[biasNeuronName] = ([], biasNeuronModel)    #add new bias neuron to connections dictionary
            bias_implementation[0] = biasNeuronName        #update bias_implementation with new current bias neuron name
        connections[biasNeuronName][0].append((neuronName, bias))    
        
    # Handler implementations (accepting standardized parameters)
    def _convert_conv2d(self, **kwargs):
        """
        Convert an nn.Conv2d layer into CRI neurons and synapses.

        One neuron is created per output pixel per output channel, named
        conv{L}.{out_channel}.{pixel} with pixel running row-major
        over the out_h x out_w output map. The presynaptic neurons and
        corresponding connections to each neuron is found by applying F.unfold
        (with the layer's kernel_size, stride and padding) to a map of 1-based
        source indices. Padding positions unfold to 0 and are skipped, so
        no synapses are created at these positions.

        Supported predecessors
            - None (first layer): input axons A1 ... A{C*H*W} are
              created and connected to the conv neurons.
            - "conv" / "maxpool": synapses are added from the
              previous layer's {type}{L-1}.{in_channel}.{pixel} neurons.

        Neuron model
            IF / LIF: threshold theta equals the layer's weight
            quantization scale, i.e. the int16 equivalent of a
            floating-point activation of 1.0. Users may also manually
            input a threshold if QAT was not used.
            ANN: theta=0

        Parameters
        ----------
        **kwargs
            Standard handler arguments (see register_handler()). Uses
            module.kernel_size, module.stride and module.padding.
            Dilation and groups other than the defaults are not supported.

        Returns
        -------
        tuple[str, int, dict[str, int]]
            ("conv", prev_layer_idx + 1, layer_counters).

        Raises
        ------
        ValueError
            If neuron_type is unsupported or the previous layer is a
            "linear" / "gap" layer.
        """
        # Conv2d specific logic using kwargs['shapes']['in_shape'], etc.
        curr_layer_type = "conv"

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
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding

        # 3. Retrieve weights, scale,bias, and neuron type for this specific layer name
        weight_key = f"{layer_name}.weight"
        bias_key = f"{layer_name}.bias"
        bias_tracker = {}
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

            #patch_rows is a tensor where #rows = resolution of the output channel feature maps
            #Each row contains the indices of the axons corresponding to each pixel in output channel feature maps
            patch_rows = patchTensor.transpose(1, 2).squeeze(0) 
            patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor
            # for each axon index in patch_rows, create a connection to the corresponding C0 neurons with the appropriate weight from layer_weights
            for output_pixel , row in enumerate(patch_rows, start=1):
                for i, elem in enumerate(row):   #each elem in row is the axon index
                    axon_id = int(elem.item())
                    if axon_id != 0:     #avoid all zeros from padding 
                        key = f"A{axon_id}"     
                        #each axon index has a connection to one neuron in each output channels of the current layer. Iterate through each output channel and create an axonal synapse to the corresponding neuron in that output channel.
                        for output_channel, kernel in enumerate(layer_weights, start=1):
                            neuronName = f"{curr_layer_type}{prev_layer_idx+1}.{output_channel}.{output_pixel}"  #Create neuron entry
                            if neuronName not in connections:  #first time this neuron is being added, initialize its connection entry
                                connections[neuronName] = ([], neuron) 
                                # first time this neuron is being added, implement bias if applicable
                                if layer_bias is not None:
                                    self.convert_bias_helper(connections, axons, layer_bias, output_channel, neuronName, bias, bias_tracker) 
                                # for the first time this neuron is being added, check if it is the output layer and add to outputs list if so
                                if layer_name == self.output_layer_name:   #if this is the output layer, add neurons to outputs list
                                    outputs.append(neuronName)
                            flat_kernel = kernel.flatten()
                            weight = flat_kernel[i].item()
                            axons[key].append((neuronName, weight))
        
        # 5. Create new layer nodes for this Conv2d layer
        elif prev_layer_type in ["conv", "maxpool"]:
            #creating cMap to identify which conv neuron from the previous layer is connected to which conv neuron in the current layer.
            cMap = torch.arange(in_h * in_w, dtype=torch.float32).reshape(1, 1, in_h, in_w)
            cMap = cMap + 1          
            patchTensor = F.unfold(input=cMap, kernel_size=kernel_size, stride=stride, padding=padding)   # dilation=1 by default

            #patch_rows is a tensor where #rows = resolution of output channel feature maps.
            #Each row contains the indices of the axons corresponding to each pixel in the output channel feature maps
            patch_rows = patchTensor.transpose(1, 2).squeeze(0)
            patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor
                    
            for output_pixel, row in enumerate(patch_rows, start=1):
                for i, elem in enumerate(row):
                    input_pixel = int(elem.item())
                    if input_pixel != 0:
                        for output_channel, output_filters in enumerate(layer_weights, start=1):  #each output channel has an output filter consisting of x number of kernels where x = number of input channels
                            neuronName = f"{curr_layer_type}{prev_layer_idx+1}.{output_channel}.{output_pixel}" 
                            if neuronName not in connections:  #first time this neuron is being added, initialize its connection entry
                                connections[neuronName] = ([], neuron) 
                                #this is first time this neuron is being added, implement bias if applicable
                                if layer_bias is not None:
                                    self.convert_bias_helper(connections, axons, layer_bias, output_channel, neuronName, bias, bias_tracker) 
                                #this is first time this neuron is being added, check if it is the output layer and add to outputs list if so
                                if layer_name == self.output_layer_name:   
                                    outputs.append(neuronName)
                                    
                            #iterate over the kernels in each output filter. The number of kernels equals the number of input channels.
                            for input_channel, kernel in enumerate(output_filters, start=1):
                                key = f"{prev_layer_type}{prev_layer_idx}.{input_channel}.{input_pixel}"
                                flat_kernel = kernel.flatten() #flatten kernel is 1D tensor of the weights for C1 -> C2
                                weight = flat_kernel[i].item()
                                connections[key][0].append((neuronName, weight))

        else:
            raise ValueError(f"Unsupported previous layer type: {prev_layer_type} for layer {layer_name}")

        prev_layer_type = curr_layer_type
        prev_layer_idx += 1
        layer_counters['conv'] += 1
    
        return prev_layer_type, prev_layer_idx, layer_counters

    def _convert_linear(self, **kwargs):
        """
        Convert an nn.Linear layer into CRI neurons and synapses.

        One neuron is created per output feature j, named linear{L}.{j}.
        Weight W[j, i] becomes a synapse from input/source i to
        neuron j.

        Supported predecessors
            - None (first layer): input axons A1 ... A{in_features}
              are created and connected to the linear neurons.
            - "conv" / "maxpool": the input is assumed to be the
              previous layer's output flattened in (C, H, W) order, so input
              column col maps to neuron
              {type}{L-1}.{col // (H*W) + 1}.{col % (H*W) + 1}. +1 due to 1-based indexing
            - "linear" / "gap": input column col maps to neuron
              {type}{L-1}.{col + 1}. +1 due to 1-based indexing

        Neuron model
            Same rules as _convert_conv2d().

        Parameters
        ----------
        **kwargs
            Standard handler arguments (see register_handler()).

        Returns
        -------
        tuple[str, int, dict[str, int]]
            ("linear", prev_layer_idx + 1, layer_counters).

        Raises
        ------
        ValueError
            If neuron_type or the previous layer type is unsupported, or
            if the traced input/output shapes are not 2-D
            (batch, features).
        """
        # Linear logic
        curr_layer_type = "linear"

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
        layer_name = kwargs['layer_name']
        neuron_type = kwargs['neuron_type']
        bias = kwargs['bias']  # Bias neuron for this layer (if not in energy-efficient mode)

        # 2. Extract spatial metadata automatically from `shapes`
        in_h, in_w = shapes['in_shape']

        # 3. Retrieve weights, scale,bias, and neuron type for this specific layer name
        weight_key = f"{layer_name}.weight"
        bias_key = f"{layer_name}.bias"
        bias_tracker = {}
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
                for output_feature, weight in enumerate(layer_weights[:, i], start=1):
                    neuronName = f"{curr_layer_type}{prev_layer_idx+1}.{output_feature}"
                    if neuronName not in connections:  #first time this neuron is being added, initialize its connection entry
                        connections[neuronName] = ([], neuron)
                        if layer_bias is not None:
                            self.convert_bias_helper(connections, axons, layer_bias, output_feature, neuronName, bias, bias_tracker) 
                    connectingNeuron = (neuronName, weight.item())
                    allConnections.append(connectingNeuron)
                axons[f"A{i+1}"] = allConnections

            if layer_name == self.output_layer_name:   #if this is the last layer, add neurons to outputs list
                for neuron in allConnections:
                    outputs.append(neuron[0])  #add neuronName to outputs list
            
        # 5. Check if prev_layer_type is "conv", "linear", "GAP", or "MaxPool" to determine how to connect the previous layer's neurons to this linear layer's neurons
        elif prev_layer_type in ["conv", "maxpool"]:
            # connect conv neurons to linear neurons
            _, _, prev_in_h, prev_in_w = kwargs['shapes'][prev_layer_name]['out_shape']  #extract output shape of previous conv layer
            feature_map = 1
            for col in range(layer_weights.shape[1]):  #x.shape[1] == number of col
                allConnections = []
                if col % (prev_in_h * prev_in_w) == 0 and col != 0:  
                    feature_map += 1
                for output_feature, elem in enumerate(layer_weights[:, col], start=1):     #iterate over element in a col
                    neuronName = f"{curr_layer_type}{prev_layer_idx+1}.{output_feature}"
                    if neuronName not in connections:
                        connections[neuronName] = ([], neuron)
                        if layer_bias is not None:
                            self.convert_bias_helper(connections, axons, layer_bias, output_feature, neuronName, bias, bias_tracker)
                    connectingNeuron = (neuronName, elem.item())
                    allConnections.append(connectingNeuron)

                for connectingNeuron in allConnections:
                    connections[f"{prev_layer_type}{prev_layer_idx}.{feature_map}.{(col % (prev_in_h * prev_in_w)) + 1}"][0].append(connectingNeuron)

            if layer_name == self.output_layer_name:   #if this is the last layer, add neurons to outputs list
                for neuron in allConnections:
                    outputs.append(neuron[0])  #add neuronName to outputs list
            
        elif prev_layer_type in ["linear", "gap"]:
            # connect linear neurons to linear neurons
            for col in range(layer_weights.shape[1]):  #x.shape[1] == number of col
                allConnections = []
                for output_feature, elem in enumerate(layer_weights[:, col], start=1):     #iterate over element in a col
                    neuronName = f"{curr_layer_type}{prev_layer_idx+1}.{output_feature}"
                    if neuronName not in connections:
                        connections[neuronName] = ([], neuron)
                        if layer_bias is not None:
                            self.convert_bias_helper(connections, axons, layer_bias, output_feature, neuronName, bias, bias_tracker)
                    connectingNeuron = (neuronName, elem.item())
                    allConnections.append(connectingNeuron)

                for connectingNeuron in allConnections:
                    connections[f"{prev_layer_type}{prev_layer_idx}.{col+1}"][0].append(connectingNeuron)

            if layer_name == self.output_layer_name:   #if this is the last layer, add neurons to outputs list
                for neuron in allConnections:
                    outputs.append(neuron[0])  #add neuronName to outputs list

        else:
            raise ValueError(f"Unsupported previous layer type: {prev_layer_type} for layer {layer_name}")
    
        prev_layer_type = curr_layer_type
        prev_layer_idx += 1
        layer_counters['linear'] += 1
            
        return prev_layer_type, prev_layer_idx, layer_counters

    def _convert_gap(self, **kwargs):
        """
        Convert a global nn.AdaptiveAvgPool2d layer into CRI neurons.

        One neuron is created per input channel, named gap{L}.{channel}.
        Every pixel of that channel connects to it with weight 1, so the
        neuron accumulates the channel *sum*. The division by H*W is
        handled by the threshold: IF / LIF neurons get theta = H*W. 

        ****Implementation of GAP for ANN neurons is untested/unverified***

        Supported predecessors
            - None (first layer): input axons are created and connected
              to the GAP neurons.
            - "conv" / "maxpool".

        Parameters
        ----------
        **kwargs
            Standard handler arguments (see register_handler()). The
            weights, scales and bias arguments are ignored because pooling
            has no parameters.

        Returns
        -------
        tuple[str, int, dict[str, int]]
            ("gap", prev_layer_idx + 1, layer_counters).

        Raises
        ------
        ValueError
            If the layer's output is not (C, 1, 1) with the same number
            of channels as its input (only *global* average pooling is
            supported), if neuron_type is unsupported, or if the previous
            layer is "linear" / "gap".
        """
        # Global Average Pooling logic
        curr_layer_type = "gap"

         # 1. Extract only the variables THIS layer needs from kwargs
        layer_name = kwargs['layer_name']
        shapes = kwargs['shapes'][layer_name]               # From forward hook tracer
        connections = kwargs['connections']     # Global CRI connections dict
        axons = kwargs['axons']                   # Global CRI axons dict
        outputs = kwargs['outputs']               # Global CRI outputs list
        prev_layer_type = kwargs['prev_layer_type']
        prev_layer_idx = kwargs['prev_layer_idx']
        layer_counters = kwargs['layer_counters']
        neuron_type = kwargs['neuron_type']

        # 2. Extract spatial metadata automatically from `shapes`
        _, in_c, in_h, in_w = shapes['in_shape']
        _, out_c, out_h, out_w = shapes['out_shape']

        if out_h != 1 or out_w != 1 or out_c != in_c:
            raise ValueError(f"Global Average Pooling layer {layer_name} must have output size of (1, 1), but got ({out_h}, {out_w}). Only GAP conversion is supported in this converter.")

        if neuron_type == "IF":
            neuron = IF_neuron(theta=(in_h * in_w))  # MaxPool neurons typically don't have learnable parameters, so we can set a default threshold

        elif neuron_type == "LIF":
            neuron = LIF_neuron(theta=(in_h * in_w))  # Similar reasoning as above

        elif neuron_type == "ANN":
            neuron = ANN_neuron(theta=0)

        else:
            raise ValueError(f"Unsupported neuron type detected: {neuron_type} for layer {layer_name}")


        print(f"Converting GAP layer: {layer_name}")
        print(f"input shape: {shapes['in_shape']}, output shape: {shapes['out_shape']}")
        # 3. Check if prev_layer_type is None; if so, this is the first layer. Create axons and connections accordingly.
        if prev_layer_type is None:
            # Create input axons for the first layer
            for i in range(1, (in_c * in_h * in_w) + 1): # Axon indices start from 1
                axons[f"A{i}"] = []

            # axons -> GAPneurons
            # The GAP window is the whole feature map, so every axon of channel x connects to GAP neuron x. Note: all weights are 1
            for x in range(1, (in_c + 1)): #iterate through each input channel
                neuronName = f"{curr_layer_type}{prev_layer_idx+1}.{x}"  #Create neuron entry GAP1.{input_channel}
                if neuronName not in connections:  #first time this neuron is being added, initialize its connection entry
                    connections[neuronName] = ([], neuron)
                    if layer_name == self.output_layer_name:   #if this is the output layer, add neurons to outputs list
                        outputs.append(neuronName)
                for i in range(1, (in_h * in_w) + 1): #iterate through each pixel in the input channel feature map
                    axon_id = (x - 1) * in_h * in_w + i   #axons are numbered 1-based in (C, H, W) order
                    axons[f"A{axon_id}"].append((neuronName, 1)) #weight is always 1 for GAP connections

        elif prev_layer_type in ["conv", "maxpool"]:
            for x in range(1, (in_c + 1)): #iterate through each input channel
                for i in range(1, (in_h * in_w) + 1): #iterate through each neuron in feature map
                    neuronName = f"{prev_layer_type}{prev_layer_idx}.{x}.{i}"
                    GAP_neuron_name = f"{curr_layer_type}{prev_layer_idx+1}.{x}"  #Create neuron entry GAP1.{input_channel}
                    if GAP_neuron_name not in connections:  #first time this neuron is being added, initialize its connection entry
                        connections[GAP_neuron_name] = ([], neuron) 
                        if layer_name == self.output_layer_name:   #if this is the output layer, add neurons to outputs list
                            outputs.append(GAP_neuron_name)
                    connections[neuronName][0].append((GAP_neuron_name, 1)) #connect each neuron in the input channel feature map to one GAP neuron with weight of 1

        else:
            raise ValueError(f"Unsupported previous layer type: {prev_layer_type} for layer {layer_name}")

        prev_layer_type = curr_layer_type
        prev_layer_idx += 1
        layer_counters['gap'] += 1
        return prev_layer_type, prev_layer_idx, layer_counters

    def _convert_maxpool(self, **kwargs):
        """
        Convert an nn.MaxPool2d layer into CRI neurons.

        One neuron is created per output pixel per channel, named
        maxpool{L}.{channel}.{pixel}. Each maxpool neuron receives a synapse of
        weight 1 from every neuron in its pooling window on the *same*
        channel (window found with F.unfold as in _convert_conv2d()). Recall 
        the number of input channels equals the number of output channels for a maxpool
        layer.

        ****Implementation of MaxPool for IF and LIF neurons is untested/unverified***
        
        With maxpool ANN neurons (theta=0) the neuron activates as long as one of the
        neurons in its pooling window activates

        Supported predecessors
            - None (first layer): input axons are created and connected
              to the max-pool neurons.
            - "conv" / "maxpool".

        Parameters
        ----------
        **kwargs
            Standard handler arguments (see register_handler()). Uses
            module.kernel_size, module.stride and module.padding.
            The weights, scales and bias arguments are ignored.

        Returns
        -------
        tuple[str, int, dict[str, int]]
            ("maxpool", prev_layer_idx + 1, layer_counters).

        Raises
        ------
        ValueError
            If neuron_type is unsupported or if the previous
            layer is "linear" / "gap".
        """
        # Max Pooling logic
        curr_layer_type = "maxpool"
        
        # 1. Extract only the variables THIS layer needs from kwargs
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

        # 2. Extract spatial metadata automatically from `module` and `shapes`
        _, in_c, in_h, in_w = shapes['in_shape']
        kernel_size = module.kernel_size
        stride = module.stride
        padding = module.padding

        if neuron_type == "IF":
            neuron = IF_neuron(theta=1)  # MaxPool neurons typically don't have learnable parameters, so we can set a default threshold

        elif neuron_type == "LIF":
            neuron = LIF_neuron(theta=1)  # Similar reasoning as above

        elif neuron_type == "ANN":
            neuron = ANN_neuron(theta=0)

        else:
            raise ValueError(f"Unsupported neuron type detected: {neuron_type} for layer {layer_name}")
        
        print(f"Converting MaxPool layer: {layer_name} with kernel_size={kernel_size}, stride={stride}, padding={padding}")
        print(f"input shape: {shapes['in_shape']}, output shape: {shapes['out_shape']}")
        # 3. Check if prev_layer_type is None; if so, this is the first layer. Create axons and connections accordingly.
        if prev_layer_type is None:
            # Create input axons for the first layer
            for i in range(1, (in_c * in_h * in_w) + 1): # Axon indices start from 1
                axons[f"A{i}"] = []

            # axons -> MaxPoolneurons
            axonMap = torch.arange(in_c * in_h * in_w, dtype=torch.float32).reshape(1, in_c, in_h, in_w)
            axonMap = axonMap + 1 #each entry labeled 1 to in_c * in_h * in_w to ignore zeros from padding          
            patchTensor = F.unfold(input=axonMap, kernel_size=kernel_size, stride=stride, padding=padding)   

            #patch_rows is a tensor where #rows = resolution of output channel feature maps
            #Each row contains the indices of the axons corresponding to each pixel in the feature map
            patch_rows = patchTensor.transpose(1, 2).squeeze(0) 
            patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor
            # for each axon index in patch_rows, create a connection to the corresponding MaxPool neurons. Note: all weights are 1
            for output_pixel , row in enumerate(patch_rows, start=1):
                for i, elem in enumerate(row):   #each elem in row is the axon index
                    axon_id = int(elem.item())
                    if axon_id != 0:     #avoid all zeros from padding 
                        key = f"A{axon_id}"     
                        for channel in range (1, (in_c)+1):  #iterate through the number of input channels
                            neuronName = f"{curr_layer_type}{prev_layer_idx+1}.{channel}.{output_pixel}"  #Create neuron entry MAXPOOL1.{channel}.{output_pixel}
                            if neuronName not in connections:  #first time this neuron is being added, initialize its connection entry
                                connections[neuronName] = ([], neuron) 
                                if layer_name == self.output_layer_name:   #if this is the output layer, add neurons to outputs list
                                    outputs.append(neuronName)
                            axons[key].append((neuronName, 1)) #weight is always 1 for MaxPool connections
        
        # 4. Create new layer nodes for MaxPool layer
        elif prev_layer_type in ["conv", "maxpool"]:
            #creating maxPoolMap to identify which neuron from the previous layer is connected to which maxpool neuron in the current layer.
            MaxpoolMap = torch.arange(in_h * in_w, dtype=torch.float32).reshape(1, 1, in_h, in_w)
            MaxpoolMap = MaxpoolMap + 1          
            patchTensor = F.unfold(input=MaxpoolMap, kernel_size=kernel_size, stride=stride, padding=padding)   # dilation=1 by default

            #patch_rows is a tensor where #rows = resolution of output channel feature maps.
            #Each row contains the indices of the neurons corresponding to each pixel in the output feature map
            patch_rows = patchTensor.transpose(1, 2).squeeze(0)
            patch_rows = patch_rows.to(torch.int16)   #convert patch_rows from FP32 tensor to INT16 tensor
                    
            for output_pixel, row in enumerate(patch_rows, start=1):
                for i, elem in enumerate(row):
                    input_pixel = int(elem.item())
                    if input_pixel != 0:
                        for channel in range(1, (in_c)+1):  #iterate through number of input channels
                            key = f"{prev_layer_type}{prev_layer_idx}.{channel}.{input_pixel}"
                            neuronName = f"{curr_layer_type}{prev_layer_idx+1}.{channel}.{output_pixel}" 
                            if neuronName not in connections:  #first time this neuron is being added, initialize its connection entry
                                connections[neuronName] = ([], neuron)
                                if layer_name == self.output_layer_name:   
                                    outputs.append(neuronName)
                            connections[key][0].append((neuronName, 1))

        else:
            raise ValueError(f"Unsupported previous layer type: {prev_layer_type} for layer {layer_name}")

        prev_layer_type = "maxpool"
        prev_layer_idx += 1
        layer_counters['maxpool'] += 1

        return prev_layer_type, prev_layer_idx, layer_counters