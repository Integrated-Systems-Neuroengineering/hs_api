import sys
sys.path.insert(0, '/home/prpandit/hs_api')
sys.path.insert(0, '/home/prpandit/hs_bridge')
sys.path.insert(0, '/home/prpandit/connectome_utils')

import pickle
from hs_api.api import CRI_network
from hs_bridge.partitioning_code.hierarchical_partition import partition as netPart

with open('/home/prpandit/hs_api/tests/fixtures/DVS_model_small_config_shift=-17.pkl', 'rb') as f:
    axons, connections, outputs = pickle.load(f)

# Build connectome via simpleSim (no hardware needed)
net = CRI_network(axons=axons, connections=connections, outputs=outputs, target="simpleSim")

data = net.connectome.get_part_format()
print(f"Network has {len(data)} neurons")
membership = netPart(data=data, n_clusters=16)
print(f"Partition done: {len(set(membership.values()))} cores used")
print(f"Core distribution: {sorted([(k, list(membership.values()).count(k)) for k in set(membership.values())])}")