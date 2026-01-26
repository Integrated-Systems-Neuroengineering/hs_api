import pytest
from hs_api.api import CRI_network
from hs_api.neuron_models import ANN_neuron, LIF_neuron

class TestBitStream:
    """Test suite using pytest framework"""
    
    @pytest.fixture
    def setup_dictionaries(self):
        """Factory fixture that creates network configurations"""
        created_networks = []
        
        def _setup(numberAxons, numberNeurons, weight, neuron_model):
            """creates the axon and connection dictionaries before tests involving 1 layer of neurons"""
            
            #define dictionaries
            axons = {}
            connections = {}

            #check number of weights equals number of axons
            if isinstance(weight, list):
                assert len(weight) == numberAxons, "Length of weight list must equal number of axons"
            else:
                weight = [weight] * numberAxons  # converts weight to a list by repeating that weight for every axon

            #creating axonal connections to neuron
            for i in range(numberAxons): #connect each axon with each neuron
                axonToNeuron = []
                for j in range(numberNeurons):  
                    connectingNeuron = (f"N{j}", weight[i])
                    axonToNeuron.append(connectingNeuron)
                axons[f"A{i}"] = axonToNeuron

            #creating output neurons
            outputs = []
            for i in range(numberNeurons):    #add each neuron to the connections dictionary
                connections[f"N{i}"] = ([], neuron_model)
                outputs.append(f"N{i}")

            #create inputs list
            inputs = []
            for i in range(numberAxons):
                inputs.append(f"A{i}")

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

        def _setup(numberAxons, numberN1, numberN2, weightAxon_N1, weightN1_N2, neuron_model1, neuron_model2):
            """creates the axon and connection dictionaries before tests involving 2 layers of neurons"""
            axons = {}
            connections = {}

            #creating inputs, axons, and N1 neurons
            inputs = []
            for i in range(numberAxons): #connect each axon with each neuron
                axonToNeuron = []
                for j in range (numberN1):  
                    connections[f"N1.{j}"] = ([], neuron_model1)  #create N1 neuron
                    connectingNeuron = (f"N1.{j}", weightAxon_N1) 
                    axonToNeuron.append(connectingNeuron)
                axons[f"A{i}"] = axonToNeuron
                inputs.append(f"A{i}")

            #creating all N1 neurons and connect to N2 neurons
            for i in range(numberN1):
                for j in range(numberN2):
                    connections[f"N2.{j}"] = ([], neuron_model2)  #create N2 neuron
                    connections[f"N1.{i}"][0].append((f"N2.{j}", weightN1_N2)) #connect N1 --> N2

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
            neuron_model=ANN_neuron(0, shift=0)
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
            neuron_model=ANN_neuron(0, shift=0)
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
            neuron_model1=ANN_neuron(-1, shift=0), 
            neuron_model2=ANN_neuron(0, shift=0)
        )

        _ = network.step([]) #0th time step
        results1 = network.read_membrane(outputs)
        _ = network.step([]) #1st time step
        results2 = network.read_membrane(outputs)
        _ = network.step([]) #2nd time step
        results3 = network.read_membrane(outputs)

        #check membrane potentials
        assert results1[0] == ("N1.0", 0), "Membrane potential of N1.0 at 0th timestep does not match expected value"
        assert results1[1] == ("N2.0", 1), "Membrane potential of N2.0 at 0th timestep does not match expected value"
        assert results2[0] == ("N1.0", 0), "Membrane potential of N1.0 at 1st timestep does not match expected value"
        assert results2[1] == ("N2.0", 1), "Membrane potential of N2.0 at 1st timestep does not match expected value"
        assert results3[0] == ("N1.0", 0), "Membrane potential of N1.0 at 2nd timestep does not match expected value"
        assert results3[1] == ("N2.0", 1), "Membrane potential of N2.0 at 2nd timestep does not match expected value"

    def test_LIF_neuron_negative_input(self, setup_dictionaries):
        """Test LIF neuron with negative input weights. 3 axons connected to 1 neuron

        Test Description:
        Validates that LIF neurons correctly process both positive and negative
        synaptic weights from input axons.
        
        Network Configuration:
            - 3 axons: A0 (weight=0), A1 (weight=-1), A2 (weight=+1)
            - 1 LIF neuron with threshold=20, leak=63, shift=0 (leak=63 means LIF_neuron has basically no leak)
            
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
            neuron_model=LIF_neuron(threshold=20, shift=0, leak=63)
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

    def test_spike_readout(self, setup_dictionaries_2layers):
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
        for numberN1_neurons in range(1, 4097):       #max axonal fanout is 4096
            network, inputs, _ = setup_dictionaries_2layers(
                numberAxons=1, 
                numberN1=numberN1_neurons, 
                numberN2=1, 
                weightAxon_N1=1, 
                weightN1_N2=1, 
                neuron_model1=ANN_neuron(0, shift=0), 
                neuron_model2=ANN_neuron(0, shift=0)
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

    @pytest.mark.parametrize("shift", [-17, 0])
    def test_LIF_neuron_no_noise(self, setup_dictionaries, shift):
        """Test that LIF neurons with specific shift values produce no noise.
        
        Test Description:
            Verifies that LIF neurons with shift values of -17 and 0 maintain zero
            membrane potential when no input is applied. These shift values should
            produce no noise in the neuron's membrane potential.
            
        Network Configuration:
            - 1 axon (A0) with weight=0 (dummy axon, provides no actual input)
            - 1 LIF neuron with threshold=0, shift=[parametrized: -17 or 0]
            
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
            neuron_model=LIF_neuron(threshold=0, shift=shift, leak=63)
        )
        
        # Run for 100 timesteps and verify MP is 0 at each step
        for timestep in range(100):
            network.step([])  # No input
            mp = network.read_membrane(outputs)
            assert mp[0][1] == 0, f"Shift={shift}, Timestep={timestep}: Expected MP=0, got {mp[0][1]}"

    @pytest.mark.parametrize("shift", [-1, -16, 15])
    def test_LIF_neuron_with_noise(self, setup_dictionaries, shift):
        """Test that LIF neurons with specific shift values produce noise.
        
        Test Description:
            Verifies that LIF neurons with shift values other than -17 and 0 generate
            noise in their membrane potential without axonal input.
            
        Network Configuration:
            - 1 axon (A0) with weight=0 (dummy axon, provides no actual input)
            - 1 LIF neuron with threshold=0, shift=[parametrized: -1, -16, or 15]

        Test Procedure:
            Run 100 time steps with no input spikes, accumulate total membrane potential
            
        Expected Behavior:
            Sum of MPs over 100 timesteps > 0 (noise present)
            
        Explanation:
            For shift values other than -17 and 0, the LIF neuron should inject
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
            neuron_model=LIF_neuron(threshold=0, shift=shift, leak=63)
        )
        
        # Run for 100 time steps and collect membrane potentials
        mp_sum = 0
        for timestep in range(100):
            network.step([])  # No input
            mp = network.read_membrane(outputs)
            mp_sum += mp[0][1]  # Add MP value to sum
        
        # Test passes if total MP sum is 0
        assert mp_sum > 0, f"Shift={shift}: Expected MP sum to be nonzero, got {mp_sum}"

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
            neuron_model=ANN_neuron(0, shift=0)
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
            neuron_model=ANN_neuron(1, shift=0)
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

if __name__ == '__main__':
    pytest.main([__file__, '-v'])