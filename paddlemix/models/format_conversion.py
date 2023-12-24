import sys,os
# sys.path.insert(0, '/home/aistudio/external-libraries')
# sys.path.insert(0, '/home/aistudio/work/PaddleMIX/paddlemix')
# sys.path.insert(0, '/home/aistudio/work/PaddleMIX/')
# sys.path.append('/home/aistudio/work/PaddleMIX/paddlemix')

os.environ["FLAGS_use_cuda_managed_memory"] = "true"

import paddle
import torch
from paddlenlp.transformers import LlamaForCausalLM,opt


def merge(vicuna_path, trimmed_path, save_path):
    model_dict = {}
    # load the second item: vicuna
    llama_model = LlamaForCausalLM.from_pretrained(vicuna_path, convert_from_torch=True)


    for n, p in llama_model.named_parameters():
        new_name = "language_model." + n
        model_dict[new_name] = p
    print("[1/2] load vicuna(llama typel) done!")
    
    minigpt4_state_dict = torch.load(trimmed_path)
    # load the third item: minigpt4
    for n, p in minigpt4_state_dict["model"].items():
        if n.startswith("llama_model.model"):
            new_name = n.replace("llama_model.model", "language_model.llama")
            new_p = paddle.to_tensor(p.cpu().numpy())
            model_dict[new_name] = new_p

        if n.startswith("llama_proj"):
            new_name = n.replace("llama_proj", "language_projection")
            if n.endswith("weight"):
                new_p = paddle.to_tensor(p.cpu().numpy()).transpose([1, 0])
            else:
                new_p = paddle.to_tensor(p.cpu().numpy())
            model_dict[new_name] = new_p

    print("[2/2] load language_projection, some llama weights from minigpt4 done!")

    save_path = os.path.join(save_path, "model_state.pdparams")
    paddle.save(model_dict, save_path)
    print("The checkpoint of vicuna_7b has been saved to :{}".format(save_path))
    
if __name__ == "__main__":
    vicuna_path = "/home/aistudio/data/data250423/vicuna7b-V1.1"
    trimmed_path = "/home/aistudio/data/data250290/instruct_blip_vicuna7b_trimmed.pth"
    save_path = "/home/aistudio/data"
    merge(vicuna_path, trimmed_path, save_path)