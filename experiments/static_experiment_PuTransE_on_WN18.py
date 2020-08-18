from pathlib import Path
import sys

openke_path = Path.cwd().parents[0]
print(openke_path)
sys.path.append(str(openke_path))

from openke.config import Parallel_Universe_Config
from openke.data import TrainDataLoader, TestDataLoader
from openke.module.model import TransE, TransH
import torch

if __name__ == '__main__':
    # Initialize random seed to make experiments reproducable
    # init_random_seed = randint(0, 2147483647)
    init_random_seed = 4

    print("Initial random seed is:", init_random_seed)

    # Initialize TrainDataLoader for sampling of examples
    train_dataloader = TrainDataLoader(
        in_path="../benchmarks/WN18/",
        nbatches=20,
        threads=8,
        sampling_mode="normal",
        bern_flag=0,
        filter_flag=0,
        neg_ent=1,
        neg_rel=0,
        random_seed=init_random_seed)

    # Initialize TestDataLoader which provides test data
    # test_dataloader = TestDataLoader(train_dataloader.in_path, "link")
    test_dataloader = TestDataLoader(train_dataloader.in_path, "link")

    # Set parameters for model used in the Parallel Universe Config (in this case TransE)
    param_dict = {
        'dim': 20,
        'p_norm': 1,
        'norm_flag': 1
    }

    embedding_method = TransE

    PuTransE = Parallel_Universe_Config(
        training_identifier='PuTransE_WN18',
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        initial_num_universes=5000,
        min_margin=1,
        max_margin=4,
        min_lr=0.01,
        max_lr=0.1,
        min_num_epochs=50,
        max_num_epochs=200,
        min_triple_constraint=500,
        max_triple_constraint=2000,
        min_balance=0.25,
        max_balance=0.5,
        embedding_model=embedding_method,
        embedding_model_param=param_dict,
        checkpoint_dir="../checkpoint/",
        valid_steps=100,
        save_steps=10000,
        training_setting="static",
        incremental_strategy=None)

    PuTransE.train_parallel_universes(6000)
    PuTransE.run_link_prediction()

# nbatches 50
# dim 50
#
# metric:			 MRR 		 MR 		 hit@10 	 hit@3  	 hit@1
# l(raw):			 0.092339 	 16703.464844 	 0.240800 	 0.139800 	 0.011800
# r(raw):			 0.099026 	 16703.300781 	 0.251200 	 0.154600 	 0.013200
# averaged(raw):		 0.095683 	 16703.382812 	 0.246000 	 0.147200 	 0.012500
# l(filter):		 0.113150 	 16689.511719 	 0.265000 	 0.180600 	 0.019200
# r(filter):		 0.117379 	 16690.339844 	 0.271400 	 0.188000 	 0.020600
# averaged(filter):	 0.115265 	 16689.925781 	 0.268200 	 0.184300 	 0.019900

# nbatches 50
# dim 20
#
# metric:			 MRR 		 MR 		 hit@10 	 hit@3  	 hit@1
# l(raw):			 0.121770 	 16676.818359 	 0.255600 	 0.153000 	 0.055200
# r(raw):			 0.131396 	 16642.433594 	 0.257800 	 0.163200 	 0.065600
# averaged(raw):		 0.126583 	 16659.625000 	 0.256700 	 0.158100 	 0.060400
# l(filter):		 0.151477 	 16662.998047 	 0.282600 	 0.196600 	 0.079000
# r(filter):		 0.154953 	 16629.630859 	 0.278600 	 0.197600 	 0.085000
# averaged(filter):	 0.153215 	 16646.314453 	 0.280600 	 0.197100 	 0.082000

# nbatches 100
# dim 20
#
# metric:			 MRR 		 MR 		 hit@10 	 hit@3  	 hit@1
# l(raw):			 0.123551 	 16679.789062 	 0.249200 	 0.157600 	 0.057000
# r(raw):			 0.130349 	 16653.384766 	 0.259000 	 0.163200 	 0.064600
# averaged(raw):		 0.126950 	 16666.585938 	 0.254100 	 0.160400 	 0.060800
# l(filter):		 0.154093 	 16665.988281 	 0.277600 	 0.200200 	 0.082600
# r(filter):		 0.156799 	 16640.580078 	 0.280600 	 0.196200 	 0.088600
# averaged(filter):	 0.155446 	 16653.285156 	 0.279100 	 0.198200 	 0.085600

# nbatches 100
# dim 50
#
# metric:			 MRR 		 MR 		 hit@10 	 hit@3  	 hit@1
# l(raw):			 0.089643 	 16721.835938 	 0.237000 	 0.135200 	 0.009200
# r(raw):			 0.096710 	 16688.917969 	 0.249600 	 0.150600 	 0.010200
# averaged(raw):		 0.093177 	 16705.376953 	 0.243300 	 0.142900 	 0.009700
# l(filter):		 0.110187 	 16707.863281 	 0.262200 	 0.176400 	 0.016800
# r(filter):		 0.115228 	 16675.880859 	 0.270000 	 0.187000 	 0.017800
# averaged(filter):	 0.112708 	 16691.871094 	 0.266100 	 0.181700 	 0.017300

# nbatches 10
# dim 50
#
# metric:			 MRR 		 MR 		 hit@10 	 hit@3  	 hit@1
# l(raw):			 0.091094 	 16725.181641 	 0.236000 	 0.137800 	 0.011400
# r(raw):			 0.096656 	 16702.503906 	 0.249200 	 0.147600 	 0.013000
# averaged(raw):		 0.093875 	 16713.843750 	 0.242600 	 0.142700 	 0.012200
# l(filter):		 0.111601 	 16711.263672 	 0.261400 	 0.175000 	 0.020200
# r(filter):		 0.114708 	 16689.548828 	 0.268200 	 0.182200 	 0.021200
# averaged(filter):	 0.113155 	 16700.406250 	 0.264800 	 0.178600 	 0.020700