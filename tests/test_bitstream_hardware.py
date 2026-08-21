import pytest
from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron, LIF_neuron
import pytest

#shift values for ANN neurons are 0
#shift values for LIF neurons are -17 (for tests not involving noise)
#does not have synaptic delay or refractory period tests

class TestBitStream:
    """Test suite using pytest framework"""
    
    @pytest.fixture
    def setup_dictionaries(self):
        """Factory fixture that creates network configurations"""
        created_networks = []
        
        def _setup(numberAxons, numberNeurons, weight, neuron_model, fully_connect_axons_to_neurons=True):
            """creates the axon and connection dictionaries before tests involving 1 layer of neurons"""
            
            #define dictionaries
            axons = {}
            connections = {}

            #check number of weights equals number of axons
            if isinstance(weight, list):
                assert len(weight) == numberAxons, "Length of weight list must equal number of axons"
            else:
                weight = [weight] * numberAxons  # converts weight to a list by repeating that weight for every axon

            if fully_connect_axons_to_neurons:
                #creating inputs, axons, and N1 neurons
                inputs = []

                #create N1 neurons
                for i in range(numberNeurons):
                    connections[f"N1.{i}"] = ([], neuron_model)

                #creating axonal connections to neuron
                for i in range(numberAxons): #connect each axon with each neuron
                    axonToNeuron = []
                    for j in range(numberNeurons):  
                        connectingNeuron = (f"N1.{j}", weight[i])
                        axonToNeuron.append(connectingNeuron)
                    axons[f"A{i}"] = axonToNeuron
                    inputs.append(f"A{i}")

            else:
                #creating inputs, axons, and N1 neurons (each axon connects to only one N1 neuron)
                inputs = []
                for i in range(numberAxons): #connect each axon with each neuron
                    connections[f"N1.{i}"] = ([], neuron_model)  #create N1 neuron
                    connectingNeuron = (f"N1.{i}", weight[i]) 
                    axons[f"A{i}"] = [connectingNeuron]
                    inputs.append(f"A{i}")

                
            #creating output neurons
            outputs = []
            for i in range(numberNeurons):    #add each neuron to the connections dictionary
                outputs.append(f"N1.{i}")

            network = CRI_network(axons=axons, connections=connections, outputs=outputs, target="CRI")
            created_networks.append(network)
            
            return network, inputs, outputs
        
        yield _setup  # Return the factory function
        
        # Cleanup code runs here after test
        for network in created_networks:
            pass  # Add cleanup if needed (e.g., network.cleanup())
    
    @pytest.fixture
    def setup_dictionaries_2layers(self):
        """Factory fixture that creates network configurations with 2 layers of neurons"""
        created_networks = []

        def _setup(numberAxons, numberN1, numberN2, weightAxon_N1, weightN1_N2, neuron_model1, neuron_model2, fully_connect_A1_to_N1=True, output_neurons_only_N2=True):
            """creates the axon and connection dictionaries before tests involving 2 layers of neurons"""
            axons = {}
            connections = {}

            if fully_connect_A1_to_N1:
                #creating inputs, axons, and N1 neurons
                inputs = []

                #create N1 neurons
                for i in range(numberN1):
                    connections[f"N1.{i}"] = ([], neuron_model1)  

                #connect each axon with each N1 neuron
                for i in range(numberAxons): 
                    axonToNeuron = []
                    for j in range (numberN1):  
                        connectingNeuron = (f"N1.{j}", weightAxon_N1) 
                        axonToNeuron.append(connectingNeuron)
                    axons[f"A{i}"] = axonToNeuron
                    inputs.append(f"A{i}")
            
            else:
                #creating inputs, axons, and N1 neurons (each axon connects to only one N1 neuron)
                inputs = []
                for i in range(numberAxons): #connect each axon with each neuron
                    connections[f"N1.{i}"] = ([], neuron_model1)  #create N1 neuron
                    connectingNeuron = (f"N1.{i}", weightAxon_N1) 
                    axons[f"A{i}"] = [connectingNeuron]
                    inputs.append(f"A{i}")

            #create all N2 neurons
            for j in range(numberN2):
                connections[f"N2.{j}"] = ([], neuron_model2)  #create N2 neuron

            #creating all N1 neurons and connect to N2 neurons
            for i in range(numberN1):
                for j in range(numberN2):
                    connections[f"N1.{i}"][0].append((f"N2.{j}", weightN1_N2)) #connect N1 --> N2

            if output_neurons_only_N2:
                #creating output neurons (can only read spikes from output neurons)
                outputs = []
                for i in range(numberN2):    #add N2 neurons to the output list
                    outputs.append(f"N2.{i}")
            
            else:
                #creating output neurons (can only read spikes from output neurons)
                outputs = []
                for i in range(numberN1):    #add N1 neurons to the output list
                    outputs.append(f"N1.{i}")
                for i in range(numberN2):    #add N2 neurons to the output list
                    outputs.append(f"N2.{i}")

            network = CRI_network(axons=axons,connections=connections,outputs=outputs,target="CRI")
            created_networks.append(network)
            
            return network, inputs, outputs
        
        yield _setup  # Return the factory function
        
        # Cleanup code runs here after test
        for network in created_networks:
            pass  # Add cleanup if needed (e.g., network.cleanup())

    @pytest.mark.parametrize("numberAxons", [1000, 5000, 10000, 16000, 16382, 16383, 16384])
    def test_max_number_axons(self, setup_dictionaries, numberAxons):
        """Test maximum number of axons (16383) each axon connects to 1 neuron

        Test Description:
            Verifies that the network can handle the maximum number of axons (16383)
            with each axon connecting to a single neuron.

        Network Configuration:
            - 16383 axons (A0-A16382), all with weight=1
            - 16383 ANN neuron (N0-N16382) with threshold=0, shift=0
            - Each axon connects to a unique neuron (A0->N0, A1->N1, ..., A16382->N16382)
            
        Test Procedure:
            1. Time step 0: Activate all 16383 axons
            2. Time step 1: No input
            
        Expected Behavior:
            - Time step 0: No spikes (neuron accumulates input from axons, MP=1 for each neuron)
            - Time step 1: Neuron spikes (MP > threshold), then resets to MP=0
            
        Explanation:
            ANN neurons with threshold=0 spike immediately when MP > 0. At time step 0,
            each neuron receives input from one of 16383 axons but doesn't spike until the next time step
            due to the one-cycle delay in the CRI architecture.
        """
        network, inputs, outputs = setup_dictionaries(
            numberAxons=numberAxons, 
            numberNeurons=numberAxons, 
            weight=1, 
            neuron_model=ANN_neuron(theta = 0),
            fully_connect_axons_to_neurons=False  #each axon connects to only one neuron
        )
        
        currSpikes1 = network.step(inputs) #0th time step
        mp1 = network.read_membrane(outputs)
        currSpikes2 = network.step([]) #1st time step
        mp2 = network.read_membrane(outputs)

        #convert mp1, mp2 to dictionaries for easier access
        mp1_dict = dict(mp1)
        mp2_dict = dict(mp2)

        assert len(currSpikes1[0]) == 0
        assert len(currSpikes2[0]) != 0

        for i in range(numberAxons):
             neuron_name = f"N1.{i}"
             assert mp1_dict[neuron_name] == 1, f"Membrane potential of {neuron_name} after 0th timestep does not match expected value"
             assert mp2_dict[neuron_name] == 0, f"Membrane potential of {neuron_name} after 1st timestep does not match expected value"

    def test_number_axons_multiple_256(self, setup_dictionaries):
        """Test number of axons is multiple of 256

        Test Description:
            Verifies that the network with axon counts that are multiples
            of 256 behaves as expected.

        Network Configuration:
            - 512 axons (A0-A511), all with weight=1
            - 1 ANN neuron (N0) with threshold=0, shift=0
            - All axons connect to the single neuron
            
        Test Procedure:
            1. Time step 0: Activate all 512 axons
            2. Time step 1: No input
            
        Expected Behavior:
            - Time step 0: No spikes (neuron accumulates input from axons, MP=512)
            - Time step 1: Neuron spikes (MP > threshold), then resets to MP=0
            
        Explanation:
            ANN neurons with threshold=0 spike immediately when MP > 0. At time step 0,
            the neuron receives 512 inputs but doesn't spike until the next time step
            due to the one-cycle delay in the CRI architecture.
        """
        network, inputs, outputs = setup_dictionaries(
            numberAxons=512, 
            numberNeurons=1, 
            weight=1, 
            neuron_model=ANN_neuron(theta = 0)
        )
        
        currSpikes1 = network.step(inputs) #0th time step
        mp1 = network.read_membrane(outputs)
        currSpikes2 = network.step([]) #1st time step
        mp2 = network.read_membrane(outputs)

        assert len(currSpikes1[0]) == 0
        assert len(currSpikes2[0]) != 0
        assert mp1[0][1] == 512, "Membrane potential after 0th timestep does not match expected value"
        assert mp2[0][1] == 0, "Membrane potential after 1st timestep does not match expected value"

    def test_number_axons_not_multiple_256(self, setup_dictionaries):
        """Test number of axons is not multiple of 256

        Test Description:
            Verifies that the network with axon counts that are not multiples
            of 256 behaves as expected.

        Network Configuration:
            - 513 axons (A0-A512), all with weight=1
            - 1 ANN neuron (N0) with threshold=0, shift=0
            - All axons connect to the single neuron
            
        Test Procedure:
            1. Time step 0: Activate all 513 axons
            2. Time step 1: No input
            
        Expected Behavior:
            - Time step 0: No spikes (neuron accumulates input from axons, MP=513)
            - Time step 1: Neuron spikes (MP > threshold), then resets to MP=0
            
        Explanation:
            ANN neurons with threshold=0 spike immediately when MP > 0. At time step 0,
            the neuron receives 513 inputs but doesn't spike until the next time step
            due to the one-cycle delay in the CRI architecture.
        """

        network, inputs, outputs = setup_dictionaries(
            numberAxons=513, 
            numberNeurons=1, 
            weight=1, 
            neuron_model=ANN_neuron(theta = 0)
        )
        
        currSpikes1 = network.step(inputs) #0th time step
        mp1 = network.read_membrane(outputs)
        currSpikes2 = network.step([]) #1st time step
        mp2 = network.read_membrane(outputs)

        assert len(currSpikes1[0]) == 0
        assert len(currSpikes2[0]) != 0
        assert mp1[0][1] == 513, "Membrane potential after 0th timestep does not match expected value"
        assert mp2[0][1] == 0, "Membrane potential after 1st timestep does not match expected value"

    def test_2layers_no_input(self, setup_dictionaries_2layers):
        """Test network with 2 layers of neurons. No input passed to network

        Test Description:
            Tests a simple 2-layer network (axon -> N1 -> N2) with no axonal input
            to verify that neurons with negative thresholds generate spontaneous activity
            and that this activity propagates correctly through the network layers.
            
        Network Configuration:
            - 1 axon (A0) with weight=1 to N1.0 (though not activated in this test)
            - 1 first-layer neuron (N1.0): ANN with threshold=-1, shift=0
            - 1 second-layer neuron (N2.0): ANN with threshold=0, shift=0
            - Connection: N1.0 -> N2.0 with weight=1
            
        Test Procedure:
            Run 3 time steps with no axonal input (empty spike list to network.step([]))
            Read membrane potentials after each time step
            
        Expected Behavior:
            - At each time step: N1.0 has MP=0 (spikes every cycle, then resets)
            - At each time step: N2.0 has MP=1 (receives constant input from N1.0)
            
        Explanation:
            N1.0 has threshold=-1, meaning it spikes whenever MP > -1. With no external
            input, N1.0's MP is 0, which is greater than -1, so N1.0 spikes at every single
            time step. After spiking, its MP resets to 0, creating a constant spike source.
            
            Each time N1.0 spikes, it sends a spike to N2.0 with weight=1. N2.0 has
            threshold=0, so it spikes since its MP=1 > 0. This resets the MP of N2.0 to 0. 
            In the subsequent timestep, N2.0 recieves input from N1.0 again so its MP=1
            before spiking again. Thus, N2.0 maintains MP=1 for all timesteps. 
        """
        network, _, outputs = setup_dictionaries_2layers(
            numberAxons=1, 
            numberN1=1, 
            numberN2=1, 
            weightAxon_N1=1, 
            weightN1_N2=1, 
            neuron_model1=ANN_neuron(theta = -1), 
            neuron_model2=ANN_neuron(theta = 0)
        )

        for i in range(1):    #add N1 neurons to the output list
            outputs.append(f"N1.{i}")

        _ = network.step([]) #0th time step
        mp1 = network.read_membrane(outputs)
        _ = network.step([]) #1st time step
        mp2 = network.read_membrane(outputs)
        _ = network.step([]) #2nd time step
        mp3 = network.read_membrane(outputs)

        #convert mp1, mp2, mp3 to dictionaries for easier access
        mp1_dict = dict(mp1)
        mp2_dict = dict(mp2)
        mp3_dict = dict(mp3)

        #check membrane potentials
        assert mp1_dict["N1.0"] == 0, f"Membrane potential of N1.0 at 0th timestep does not match expected value. Expected 0, got {mp1_dict['N1.0']}"
        assert mp1_dict["N2.0"] == 1, f"Membrane potential of N2.0 at 0th timestep does not match expected value. Expected 1, got {mp1_dict['N2.0']}"
        assert mp2_dict["N1.0"] == 0, f"Membrane potential of N1.0 at 1st timestep does not match expected value. Expected 0, got {mp2_dict['N1.0']}"
        assert mp2_dict["N2.0"] == 1, f"Membrane potential of N2.0 at 1st timestep does not match expected value. Expected 1, got {mp2_dict['N2.0']}"
        assert mp3_dict["N1.0"] == 0, f"Membrane potential of N1.0 at 2nd timestep does not match expected value. Expected 0, got {mp3_dict['N1.0']}"
        assert mp3_dict["N2.0"] == 1, f"Membrane potential of N2.0 at 2nd timestep does not match expected value. Expected 1, got {mp3_dict['N2.0']}"

    def test_LIF_neuron_negative_input(self, setup_dictionaries):
        """Test LIF neuron with negative input weights. 3 axons connected to 1 neuron

        Test Description:
        Validates that LIF neurons correctly process both positive and negative
        synaptic weights from input axons.
        
        Network Configuration:
            - 3 axons: A0 (weight=0), A1 (weight=-1), A2 (weight=+1)
            - 1 LIF neuron with threshold=20, leak=63, shift=-17 (leak=63 means LIF_neuron has basically no leak)
            
        Test Procedure:
            Time step 0: No input
            Time step 1: Activate A1 (negative weight)
            Time step 2: Activate A2 (positive weight)
            Time step 3-4: No input (observe behavior)
            
        Expected Behavior:
            - Time step 0: MP=0 (no input, no spikes)
            - Time step 1: MP=-1 (negative input applied, no spikes)
            - Time step 2: MP=0 (positive cancels negative, no spikes)
            - Time step 3-4: MP=0 (no spikes)
            
        Explanation:
            One LIF neuron is connected to three axons: A0 (weight=0), A1 (weight=-1), A2 (weight=+1).
            No input is passed in the 0th time step, A1 is activated in the 1st time step, and A2 is
            activated in the 2nd time step. Thus, at each time step, the membrane potentials are expected
            to be 0, -1, 0, 0, and 0 respectively, with no spikes occurring throughout. This tests the
            hardware's ability to perform summation of both positive and negative weights.
        """
        network, inputs, outputs = setup_dictionaries(
            numberAxons=3, 
            numberNeurons=1, 
            weight=[0, -1, 1],   #weight of A0 is 0, A1 is -1, A2 is 1
            neuron_model=LIF_neuron(theta=20, nu=-17, Lambda=63)
        )
        
        FPGA_Vs = []   # membrane potential trace
        FPGA_Ss = []   # spikes 

        currSpikes1 = network.step([]) #activate no axons at 0th time step
        results1 = network.read_membrane(outputs)
        FPGA_Vs.append(results1[0][1])
        FPGA_Ss.append(currSpikes1[0])

        currSpikes2 = network.step([inputs[1]]) #activate A1 at 1st time step
        results2 = network.read_membrane(outputs)
        FPGA_Vs.append(results2[0][1])
        FPGA_Ss.append(currSpikes2[0])

        currSpikes3 = network.step([inputs[2]]) #activate A2 at 2nd time step
        results3 = network.read_membrane(outputs)
        FPGA_Vs.append(results3[0][1])
        FPGA_Ss.append(currSpikes3[0])

        currSpikes4 = network.step([]) 
        results4 = network.read_membrane(outputs)
        FPGA_Vs.append(results4[0][1])
        FPGA_Ss.append(currSpikes4[0])

        currSpikes5 = network.step([]) 
        results5 = network.read_membrane(outputs)
        FPGA_Vs.append(results5[0][1])
        FPGA_Ss.append(currSpikes5[0])

        expected_Vs = [0, -1, 0, 0, 0]
        expected_Ss = [[], [], [], [], []]

        assert FPGA_Vs == expected_Vs, f"Membrane potentials do not match expected values: Outputs {FPGA_Vs}, Expected {expected_Vs}"
        assert FPGA_Ss == expected_Ss, f"Spikes do not match expected values: Outputs {FPGA_Ss}, Expected {expected_Ss}"
    
    @pytest.mark.parametrize("numberN1_neurons", [10, 25, 30, 1000, 2000, 4095, 4096])
    def test_spike_readout(self, setup_dictionaries_2layers, numberN1_neurons):
        """Test spike readout accuracy with varying axonal fanout from 1 to 4096.
        
        Test Description:
            Validates that the network correctly reads and reports spikes from all
            neurons across a wide range of axonal fanout configurations. This
            tests the spike readout mechanism and verifies proper handling of large
            numbers of simultaneous spikes. The test stops at the first failure.
            
        Network Configuration (varies each iteration):
            - 1 axon (A0) with weight=1 to all N1 neurons
            - Variable number of N1 neurons (1 to 4096): ANN with threshold=0, shift=0
            - 1 N2 neuron: ANN with threshold=0, shift=0
            - All N1 neurons connect to N2.0 with weight=1
            
        Test Procedure (for each fanout size):
            Time step 0: Activate A0
            Time step 1: No input (N1 neurons spike)
            Time step 2: No input (N2.0 spikes)
            
        Expected Behavior:
            - Time step 0: No spikes (input propagating)
            - Time step 1: All N1 neurons spike, N2.0 MP = number of N1 neurons
            - Time step 2: N2.0 spikes and resets to MP=0
            
        Explanation:
            At time step 0, A0 activates and its signal propagates to all N1 neurons.
            At time step 1, all N1 neurons (with threshold=0) spike simultaneously,
            each contributing +1 to N2.0's membrane potential. N2.0 accumulates a
            total of (numberN1_neurons * 1). At time step 2, N2.0 spikes and resets.
            This tests the maximum axonal fanout (4096) and validates that spike
            readout works correctly for all fanout sizes up to the hardware limit.
        """
        network, inputs, _ = setup_dictionaries_2layers(
            numberAxons=1, 
            numberN1=numberN1_neurons, 
            numberN2=1, 
            weightAxon_N1=1, 
            weightN1_N2=1, 
            neuron_model1=ANN_neuron(theta = 0), 
            neuron_model2=ANN_neuron(theta = 0),
            output_neurons_only_N2=False  #read spikes from both N1 and N2 layers
        )

        # Time step 0: Activate axon
        currSpikes0 = network.step(inputs)

        # Time step 1: No input, N1 neurons spike, mp1 should be numberN1_neurons
        currSpikes1 = network.step([])
        mp1 = network.read_membrane(["N2.0"])


        # Time step 2: No input, N2 neuron spikes, mp2 should be 0
        currSpikes2 = network.step([])
        mp2 = network.read_membrane(["N2.0"])

        # Verify no spikes at time step 0
        assert len(currSpikes0[0]) == 0, f"Unexpected number of spikes at time step 0: {len(currSpikes0)}, expected 0"

        # Verify spikes from all N1 neurons at time step 1
        assert len(currSpikes1[0]) == numberN1_neurons, f"Unexpected number of spikes at time step 1: {len(currSpikes1[0])}, expected {numberN1_neurons}"
        assert mp1[0][1] == numberN1_neurons, f"Unexpected membrane potential for N2.0 at time step 1: {mp1[0][1]}, expected {numberN1_neurons}"

        # Verify spike from N2 neuron at time step 2
        assert len(currSpikes2[0]) == 1, f"Unexpected number of spikes at time step 2: {len(currSpikes2[0])}, expected 1"
        assert currSpikes2[0][0] == "N2.0", f"Unexpected spike from neuron at time step 2: {currSpikes2[0][0]}, expected 'N2.0'"
        assert mp2[0][1] == 0, f"Unexpected membrane potential for N2.0 at time step 2: {mp2[0][1]}, expected 0"

    @pytest.mark.parametrize("shift", [-17])
    def test_LIF_neuron_no_noise(self, setup_dictionaries, shift):
        """Test that LIF neurons with specific shift values produce no noise.
        
        Test Description:
            Verifies that LIF neurons with shift values of -17 and 0 maintain zero
            membrane potential when no input is applied. These shift values should
            produce no noise in the neuron's membrane potential.
            
        Network Configuration:
            - 1 axon (A0) with weight=0 (dummy axon, provides no actual input)
            - 1 LIF neuron with threshold=0, shift=-17
            
        Test Procedure:
            Run 100 time steps with no input spikes
            
        Expected Behavior:
            All 100 time steps: MP=0 (no noise)
            
        Explanation:
            The shift parameter in LIF neurons controls noise injection.
            Shift values of -17 and 0 are special cases that should produce zero noise.
            With no input (weight=0) and no noise, the membrane potential must remain
            at exactly 0 for all time steps. 
        """
        # Create network with 1 neuron, 1 dummy axon
        network, inputs, outputs = setup_dictionaries(
            numberAxons=1,
            numberNeurons=1,
            weight=0,  # No actual input weight
            neuron_model=LIF_neuron(theta=0, nu=shift, Lambda=63)
        )
        
        # Run for 100 timesteps and verify MP is 0 at each step
        for timestep in range(100):
            network.step([])  # No input
            mp = network.read_membrane(outputs)
            assert mp[0][1] == 0, f"Shift={shift}, Timestep={timestep}: Expected MP=0, got {mp[0][1]}"

    @pytest.mark.parametrize("shift", [-1, -16, 15, 16, 0])
    def test_LIF_neuron_with_noise(self, setup_dictionaries, shift):
        """Test that LIF neurons with specific shift values produce noise.
        
        Test Description:
            Verifies that LIF neurons with shift values other than -17 and 0 generate
            noise in their membrane potential without axonal input.
            
        Network Configuration:
            - 1 axon (A0) with weight=0 (dummy axon, provides no actual input)
            - 1 LIF neuron with threshold=0, shift=[parametrized: -1, -16, 15, or 0]

        Test Procedure:
            Run 100 time steps with no input spikes, accumulate total membrane potential
            
        Expected Behavior:
            Sum of absolute value of MPs over 100 timesteps > 0 (noise present)
            
        Explanation:
            For shift values other than -17, the LIF neuron should inject
            noise into the membrane potential. Even with no axonal input, the 
            neuron's MP should fluctuate due to the noise injection. Over 100
            time steps, the cumulative sum should be nonzero, demonstrating that noise
            is being generated.
        """
        # Create network with 1 neuron, 1 dummy axon
        network, inputs, outputs = setup_dictionaries(
            numberAxons=1,
            numberNeurons=1,
            weight=0,  # No actual input weight
            neuron_model=LIF_neuron(theta=0, nu=shift, Lambda=63)
        )
        
        # Run for 100 time steps and collect membrane potentials
        mp_sum = 0
        for timestep in range(100):
            network.step([])  # No input
            mp = network.read_membrane(outputs)
            mp_sum += abs(mp[0][1])  # Add absolute MP value to sum
        
        # Test passes if total MP sum is 0
        assert mp_sum > 0, f"Shift={shift}: Expected MP sum to be nonzero, got {mp_sum}"

    def test_LIF_neuron_with_noise_magnitude(self):
        """Test that LIF neurons with shift=-16 produce less noise than shift=0, and shift=16 produces more noise than shift=0."""
        axons = {}
        connections = {}

        neuron_model0 = LIF_neuron(theta=0, nu=0, Lambda=63)
        neuron_model1 = LIF_neuron(theta=0, nu=-16, Lambda=63)
        neuron_model2 = LIF_neuron(theta=0, nu=16, Lambda=63)
        axons["A0"] = [("N1.0", 0), ("N1.1", 0), ("N1.2", 0)]  # Dummy axon with weight=0 to all neurons
        connections[f"N1.0"] = ([], neuron_model0)
        connections[f"N1.1"] = ([], neuron_model1)
        connections[f"N1.2"] = ([], neuron_model2)

        outputs = ["N1.0", "N1.1", "N1.2"]
        network = CRI_network(axons=axons, connections=connections, outputs=outputs, target="CRI")

        mp_diff_sums = [0, 0, 0]  # To accumulate absolute MP differences for each neuron

        # Read the first membrane potential after the first step
        network.step([])  # No input
        prev_mp = dict(network.read_membrane(outputs))

        # remaining 999 steps and accumulate differences
        for _ in range(1, 1000):
            network.step([])  # No input
            curr_mp = dict(network.read_membrane(outputs))

            mp_diff_sums[0] += abs(curr_mp["N1.0"] - prev_mp["N1.0"])
            mp_diff_sums[1] += abs(curr_mp["N1.1"] - prev_mp["N1.1"])
            mp_diff_sums[2] += abs(curr_mp["N1.2"] - prev_mp["N1.2"])

            prev_mp = curr_mp

        print(f"Shift=0 MP sum: {mp_diff_sums[0]}, Shift=-16 MP sum: {mp_diff_sums[1]}, Shift=16 MP sum: {mp_diff_sums[2]}")

        assert mp_diff_sums[0] > mp_diff_sums[1], f"Sum of absolute MPs for shift=0 should be greater than shift=-16. Got {mp_diff_sums[0]} vs {mp_diff_sums[1]}"
        assert mp_diff_sums[2] > mp_diff_sums[0], f"Sum of absolute MPs for shift=16 should be greater than shift=0. Got {mp_diff_sums[2]} vs {mp_diff_sums[0]}"

    def test_network_reset(self, setup_dictionaries):
        """Test network reset without running flash.sh by running two different networks back-to-back
        Test Description:
            Verifies that running two different networks back-to-back works correctly
            without requiring manual hardware reset (flash.sh). This tests whether
            CRI properly resets hardware state between network instances.
            
        Network Configuration:
            First network:
                - 1 axon (A0) with weight=1
                - 1 ANN neuron with threshold=0, shift=0
            Second network (different configuration):
                - 1 axon (A0) with weight=1
                - 1 ANN neuron with threshold=1, shift=0
            
        Test Procedure:
            Network 1:
                Time step 0: Activate A0
                Time step 1: Check for spike (should spike, MP=0 after)
            Network 2:
                Time step 0: Activate A0
                Time step 1: Check for no spike (should not spike, MP=1)
                
        Expected Behavior:
            Network 1: Neuron spikes at time step 1 (threshold=0, MP=1 exceeds threshold)
            Network 2: Neuron does NOT spike at time step 1 (threshold=1, MP=1 does not exceed)
            
        Explanation:
            The first network has threshold=0, so when A0 activates (adding weight=1),
            the neuron reaches MP=1 > 0 and spikes at the next time step, resetting to 0.
            The second network has threshold=1, so when A0 activates, the neuron reaches
            MP=1 but does NOT exceed threshold, remaining at MP=1 without spiking. 
            Since flash.sh was not run in between the two networks, this tests whether the CRI
            system correctly resets between different network instances.
        """
        network, inputs, outputs = setup_dictionaries(
            numberAxons=1,
            numberNeurons=1,
            weight=1,
            neuron_model=ANN_neuron(theta = 0)
        )
        
        # time step 0: Activate axon
        network.step(inputs)
        mp1_t0 = network.read_membrane(outputs)  

        # time step 1: Neuron should spike
        spikes1 = network.step([])
        mp1_t1 = network.read_membrane(outputs)


        assert len(spikes1[0]) == 1, "Neuron did not spike as expected at time step 1"
        assert mp1_t0[0][1] == 1, f"Unexpected membrane potential at time step 0. Expected 1 before spike. Output was {mp1_t0[0][1]}"
        assert mp1_t1[0][1] == 0, f"Unexpected membrane potential at time step 1. Expected 0 after spike. Output was {mp1_t1[0][1]}"

        network2, inputs2, outputs2 = setup_dictionaries(
            numberAxons=1,
            numberNeurons=1,
            weight=1,
            neuron_model=ANN_neuron(theta = 1)
        )
        
        # time step 0: Activate axon
        network2.step(inputs2)
        mp1_network2_t0 = network2.read_membrane(outputs2)

        # time step 1: Neuron should not spike
        spikes1_network2 = network2.step([])
        mp1_network2_t1 = network2.read_membrane(outputs2)

        assert len(spikes1_network2[0]) == 0, "Neuron spiked unexpectedly at time step 1"
        assert mp1_network2_t0[0][1] == 1, f"Unexpected membrane potential at time step 0. Expected 1. Output was {mp1_network2_t0[0][1]}"
        assert mp1_network2_t1[0][1] == 0, f"Unexpected membrane potential at time step 1. Expected 0. Output was {mp1_network2_t1[0][1]}"

    @pytest.mark.parametrize("numberN1_neurons", [4000, 4050, 4095, 4096, 4097])
    def test_max_axonal_fanout(self, setup_dictionaries_2layers, numberN1_neurons):
        """Test maximum axonal fanout to neuron with detailed membrane potential checks."""
        neuron_model = ANN_neuron(theta = 0)
        network, inputs, outputs = setup_dictionaries_2layers(
                numberAxons=1, 
                numberN1=numberN1_neurons, 
                numberN2=1, 
                weightAxon_N1=1, 
                weightN1_N2=1, 
                neuron_model1=neuron_model, 
                neuron_model2=neuron_model
            )
        
        for i in range(numberN1_neurons):    #add N1 neurons to the output list
            outputs.append(f"N1.{i}")
        
        currSpikes1 = network.step(inputs) #0th time step
        mp1 = network.read_membrane(outputs)
        currSpikes2 = network.step([]) #1st time step
        mp2 = network.read_membrane(outputs)
        currSpikes3 = network.step([]) #2nd time step
        mp3 = network.read_membrane(outputs)

        #convert mp1, mp2, mp3 to dictionaries for easier access
        mp1_dict = dict(mp1)
        mp2_dict = dict(mp2)
        mp3_dict = dict(mp3)

        assert len(currSpikes1[0]) == 0    #no spikes at time step 0
        assert len(currSpikes3[0]) == 1, f"N2 did not spike as expected at time step 2. Expected 1 spike, got {len(currSpikes3[0])}"

        #check membrane potentials of all N1 neurons and N2 neuron after time steps 0
        for i in range(numberN1_neurons):
            assert mp1_dict[f"N1.{i}"] == 1, f"N1.{i} membrane potential did not match expected value at time step 0. Expected 1, got {mp1_dict[f'N1.{i}']}"

        assert mp1_dict[f"N2.0"] == 0, f"N2.0 membrane potential did not match expected value at time step 0. Expected 0, got {mp1_dict[f'N2.0']}"

        #check membrane potentials of all N1 neurons and N2 Neuron after time step 1
        for i in range(numberN1_neurons):
            assert mp2_dict[f"N1.{i}"] == 0, f"N1.{i} membrane potential did not match expected value at time step 1. Expected 0, got {mp2_dict[f'N1.{i}']}"

        assert mp2_dict["N2.0"] == numberN1_neurons, f"N2.0 membrane potential did not match expected value at time step 1. Expected {numberN1_neurons}, got {mp2_dict['N2.0']}"
    
        #check membrane potentials of all N1 neurons and N2 Neuron after time step 2
        for i in range(numberN1_neurons):
            assert mp3_dict[f"N1.{i}"] == 0, f"N1.{i} membrane potential did not match expected value at time step 2. Expected 0, got {mp3_dict[f'N1.{i}']}"
        
        assert mp3_dict["N2.0"] == 0, f"N2.0 membrane potential did not match expected value at time step 2. Expected 0, got {mp3_dict['N2.0']}"
    
    @pytest.mark.parametrize("number_axons", [8190, 8191, 8193])
    def test_max_axonal_fan_in(self, setup_dictionaries, number_axons):
        """Test maximum axonal fan-in to a single neuron from different numbers of axons."""
        network, inputs, outputs = setup_dictionaries(
            numberAxons=number_axons, 
            numberNeurons=1, 
            weight=1, 
            neuron_model=ANN_neuron(theta = 1)
        )
        
        currSpikes1 = network.step(inputs) #0th time step
        mp1 = network.read_membrane(outputs)
        currSpikes2 = network.step([]) #1st time step
        mp2 = network.read_membrane(outputs)

        #convert mp1, mp2 to dictionaries for easier access
        mp1_dict = dict(mp1)
        mp2_dict = dict(mp2)

        assert len(currSpikes1[0]) == 0    #no spikes at time step 0
        assert len(currSpikes2[0]) == 1, "Neuron did not spike as expected at time step 1"

        #check membrane potentials after time steps 0 and 1
        assert mp1_dict["N1.0"] == number_axons, f"N1.0 membrane potential did not match expected value at time step 0. Expected {number_axons}, got {mp1_dict['N1.0']}"
        assert mp2_dict["N1.0"] == 0, f"N1.0 membrane potential did not match expected value at time step 1. Expected 0, got {mp2_dict['N1.0']}"

    @pytest.mark.parametrize("numberN2_neurons", [4094, 4095, 4096, 4097])
    def test_max_neuronal_fanout(self, setup_dictionaries_2layers, numberN2_neurons):
        """Test maximum neuronal fanout from one neuron to neurons in the next layer."""
        neuron_model = ANN_neuron(theta = 0)
        network, inputs, outputs = setup_dictionaries_2layers(
                numberAxons=1, 
                numberN1=1, 
                numberN2=numberN2_neurons, 
                weightAxon_N1=1, 
                weightN1_N2=1, 
                neuron_model1=neuron_model, 
                neuron_model2=neuron_model
            )
        
        for i in range(1):    #add N1 neurons to the output list
            outputs.append(f"N1.{i}")
        
        currSpikes1 = network.step(inputs) #0th time step
        mp1 = network.read_membrane(outputs)
        currSpikes2 = network.step([]) #1st time step
        mp2 = network.read_membrane(outputs)
        currSpikes3 = network.step([]) #2nd time step
        mp3 = network.read_membrane(outputs)

        #convert mp1, mp2, mp3 to dictionaries for easier access
        mp1_dict = dict(mp1)
        mp2_dict = dict(mp2)
        mp3_dict = dict(mp3)

        #check membrane potentials of N1 neuron and all N2 neurons after time steps 0
        assert mp1_dict["N1.0"] == 1, f"N1.0 membrane potential did not match expected value at time step 0. Expected 1, got {mp1_dict['N1.0']}"
        
        for i in range(numberN2_neurons):
            assert mp1_dict[f"N2.{i}"] == 0, f"N2.{i} membrane potential did not match expected value at time step 0. Expected 0, got {mp1_dict[f'N2.{i}']}"

        #check membrane potentials of N1 neuron and all N2 neurons after time step 1
        assert mp2_dict["N1.0"] == 0, f"N1.0 membrane potential did not match expected value at time step 1. Expected 0, got {mp2_dict['N1.0']}"
        
        for i in range(numberN2_neurons):
            assert mp2_dict[f"N2.{i}"] == 1, f"N2.{i} membrane potential did not match expected value at time step 1. Expected 1, got {mp2_dict[f'N2.{i}']}"

        #check membrane potentials of N1 neuron and all N2 neurons after time step 2
        assert mp3_dict["N1.0"] == 0, f"N1.0 membrane potential did not match expected value at time step 2. Expected 0, got {mp3_dict['N1.0']}"

        for i in range(numberN2_neurons):
            assert mp3_dict[f"N2.{i}"] == 0, f"N2.{i} membrane potential did not match expected value at time step 2. Expected 0, got {mp3_dict[f'N2.{i}']}"
    
    @pytest.mark.parametrize("numberN1_neurons", [10, 1000, 5000, 8158, 8159, 8160])
    def test_neuronal_fan_in(self, setup_dictionaries_2layers, numberN1_neurons):
        """Test neuronal fan-in to a single second-layer neuron from different numbers of first-layer neurons."""
        neuron_model = ANN_neuron(theta = 0)
        network, inputs, outputs = setup_dictionaries_2layers(
                numberAxons=numberN1_neurons, 
                numberN1=numberN1_neurons, 
                numberN2=1, 
                weightAxon_N1=1, 
                weightN1_N2=1, 
                neuron_model1=neuron_model, 
                neuron_model2=neuron_model,
                fully_connect_A1_to_N1=False
            )
        
        _ = network.step(inputs) #0th time step
        _ = network.step([]) #1st time step
        results2 = network.read_membrane(outputs)
        
        assert results2[0] == ("N2.0", numberN1_neurons), f"N2.0 membrane potential did not match expected value at time step 1. Expected {numberN1_neurons}, got {results2[0][1]}"

    
if __name__ == '__main__':
    pytest.main([__file__, '-v'])