import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.graph_transformer import GraphTransformerGFN
from envs.frag_mol_env import FragMolBuildingEnvContext
from tasks.train_gfn import RoboTrainer

'''
This file uses a trained model to sample/generate robots using final_dl method of GFN.
The generated robot xml files are saved into the modified log_dir folder which is available from the hps of saved_state of the trained model.
The folder path of these generated robots can be used in evaluate_robots.py to evaluate sampled robots performance. 
'''

def cycle(it):
        while True:
            for i in it:
                yield i


def main():

    # exp = 'CA'  # Change to 'GSCA' when needed

    # if exp == 'CA' or 'GSCA':
    #     from models.graph_transformer import GraphTransformerGFN
    #     from envs.frag_mol_env import FragMolBuildingEnvContext
    #     from tasks.train_gfn import RoboTrainer
    # else:
    #     from models.graph_transformer import GraphTransformerGFN
    #     from envs.frag_mol_env import FragMolBuildingEnvContext
    #     from tasks.train_gfn import RoboTrainer

    # if exp not in ['CA', 'GSCA']:
    #     raise ValueError("exp must be either 'CA' or 'GSCA'")


    # Required Inputs based on the characteristics of the run
    lower_bound = 30
    upper_bound = 100
    step_size = 10
    path = "/home/knagiredla/robonet/logs/exp_hoppersens_1_0.002_6_20_000_25_1745678777/policies"

    # path = "/home/knagiredla/robonet/logs/exp_GSCA_5_flat_base_hopper_100_000_134_1735392360/policies"

    for idx in range(lower_bound, upper_bound+1, step_size):
        
        model_path = path + f"/model_state_{idx}.pt"

        env_ctx = FragMolBuildingEnvContext(max_frags=6, num_cond_dim=32)

        # Define an instance of YourModelClass (assuming it's the class of self.model)
        model = GraphTransformerGFN(env_ctx, num_emb=128, num_layers=4)

        # Load the model state from the saved file
        saved_state = torch.load(model_path)  # Replace with the actual file path

        # print("saved_state", saved_state)

        # Update the model's state with the loaded state
        model.load_state_dict(saved_state["models_state_dict"][0], strict=False)  # Assuming the state was saved within a list
        # Access other saved information if needed
        # Eg. model_state_dict = saved_state["models_state_dict"]
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        hps = saved_state["hps"]
        it = saved_state["step"]
        new_hps = hps

        new_hps['log_dir'] = new_hps['xml_path'] + f'/gen_{it}_steps'

        if not os.path.isdir(new_hps['log_dir']):
            os.mkdir(new_hps['log_dir'])

        trainer2 = RoboTrainer(hps=new_hps, device=device)
        # print("THIS IS TRAINER", trainer2)
        # print(type(trainer2))
        final_dl = trainer2.build_validation_data_loader()

        for it, batch in zip(
                            range(new_hps["num_training_steps"], new_hps["num_training_steps"] + 1), 
                            cycle(final_dl),
                            ):
                                pass

if __name__ == "__main__":
    main()


