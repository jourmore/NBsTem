from Bio import SeqIO
import time, random, os, re
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from model import nbstemR, nbstemC
from datetime import datetime
import warnings, math, argparse
warnings.filterwarnings("ignore", category=UserWarning,
						message="torch.utils._pytree._register_pytree_node is deprecated. Please use torch.utils._pytree.register_pytree_node instead.")

now_data = datetime.now()
current_year = now_data.year
current_month = now_data.month
current_day = now_data.day

print("")
print("*"*66)
print("**", " "*60, "**")
print("**  NBsTem v.2025 Thermostability prediction for Nanobody/VHH.  **")
print("**", " "*60, "**")
print("**                 https://www.nbscal.online/                   **")
print("**                   maojun@stu.scu.edu.cn                      **")
print("*"*66)
print("")

def parameter():
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', type=str, help='Input path with fasta format. [Such as: ./in.fasta]')
	parser.add_argument('-o', type=str, default=f'Output-NBsTem-{current_year}-{current_month}-{current_day}.csv', help='Output file name when input is fasta format. [Default: "NBsTem_year_month_day.csv"')
	parser.add_argument('-t', type=str, default='QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKEREGVATILGGSTYYGDSVKGRFTISQDNAKNTVYLQMNSLKPEDTAIYYCAGSTVASTGWCSRLRPYDYHYRGQGTQVTVSS',
						help='Input one sequecne with text format. [Default: QVQLVESGGGSVQAGGSLRLSCAASGYTVSTYCMGWFRQAPGKEREGVATILGGSTYYGDSVKGRFTISQDNAKNTVYLQMNSLKPEDTAIYYCAGSTVASTGWCSRLRPYDYHYRGQGTQVTVSS]')
	parser.add_argument('-seed', type=str, default=42, help='Random seed for torch, numpy, os. [Default: 42]')
	parser.add_argument('-device', type=str, default="auto", help='Device: cpu, cuda. [Default: auto]')
	args = parser.parse_args()
	return args

args = parameter()

def seed_everything(seed: int):   
	random.seed(seed)
	os.environ['PYTHONHASHSEED'] = str(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.benchmark = True
seed_everything(args.seed)
print("== 1.Use seed:", args.seed)

if args.device == "auto":
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # 
elif args.device == "cpu":
	device = torch.device('cpu')
else:
	device = torch.device(args.device)
print("== 2.Device:", device)

from antiberty import AntiBERTyRunner
print("== 3.Loading antibody language model: AntiBERTy")
antiberty = AntiBERTyRunner()

def read_fastatext(fasta_path):
	sequences = []
	names = []
	for i, record in enumerate(SeqIO.parse(fasta_path, "fasta"), start=1):
		sequences.append(str(record.seq))
		if record.id:
			names.append(record.id)
		else:
			names.append('NB_'+str(i))
	return names, sequences

def load_model(model, path):
	model.load_state_dict(torch.load(path, map_location=device))
	model.eval()
	return model

def Q_model5():
	LLL = 512
	modelsQ = [nbstemC(input_channel=LLL, layers=[1, 1, 1, 4], num_classes=4).to(device),
			nbstemC(input_channel=LLL, layers=[1, 1, 1, 4], num_classes=4).to(device),
			nbstemC(input_channel=LLL, layers=[1, 1, 1, 4], num_classes=4).to(device),
			nbstemC(input_channel=LLL, layers=[1, 1, 1, 4], num_classes=4).to(device),
			nbstemC(input_channel=LLL, layers=[1, 1, 1, 4], num_classes=4).to(device)]
	checkpointsQ = ['./NBsTem_Q/Q_1.pt', './NBsTem_Q/Q_2.pt', './NBsTem_Q/Q_3.pt', './NBsTem_Q/Q_4.pt', './NBsTem_Q/Q_5.pt']
	netsQ = [load_model(modelsQ[i], checkpointsQ[i]) for i in range(5)]
	return netsQ

def Tm_model5():
	LLL = 512
	modelsT = [nbstemR(input_channel=LLL, layers=[1, 1, 1, 4], num_classes=1).to(device),
			nbstemR(input_channel=LLL, layers=[1, 1, 1, 4], num_classes=1).to(device),
			nbstemR(input_channel=LLL, layers=[1, 1, 1, 4], num_classes=1).to(device),
			nbstemR(input_channel=LLL, layers=[1, 1, 1, 4], num_classes=1).to(device),
			nbstemR(input_channel=LLL, layers=[1, 1, 1, 4], num_classes=1).to(device)]
	checkpointsT = ['./NBsTem_Tm/Tm_1.pt', './NBsTem_Tm/Tm_2.pt', './NBsTem_Tm/Tm_3.pt', './NBsTem_Tm/Tm_4.pt', './NBsTem_Tm/Tm_5.pt']
	netsT = [load_model(modelsT[i], checkpointsT[i]) for i in range(5)]
	return netsT

netsQ = Q_model5()
netsT = Tm_model5()


def vote_select(predictions):
    votes = {}
    for prediction in predictions:
        # 确保提取tensor中的数值并转移到CPU
        if isinstance(prediction, torch.Tensor):
            # 使用.item()获取标量值，如果tensor在GPU上会自动转移到CPU
            pred_value = prediction.item()
        else:
            pred_value = prediction

        # 使用提取的数值进行投票
        if pred_value not in votes:
            votes[pred_value] = 0

        votes[pred_value] += 1

    final_select = max(votes, key=votes.get)
    
    return final_select

def vote_select_with_uncertainty(predictions):
    """包装函数，调用原函数并添加不确定度"""
    # 调用原函数
    final_select = vote_select(predictions)
    
    # 计算不确定度
    if not predictions:
        uncertainty = 1.0
    else:
        votes = {}
        for prediction in predictions:
            # 同样需要处理tensor设备问题
            if isinstance(prediction, torch.Tensor):
                pred_value = prediction.item()
            else:
                pred_value = prediction
                
            votes[pred_value] = votes.get(pred_value, 0) + 1
        
        total_votes = len(predictions)
        
        # 计算信息熵
        entropy = 0.0
        for count in votes.values():
            p = count / total_votes
            entropy -= p * math.log(p + 1e-8)
        
        # 归一化
        max_entropy = math.log(len(votes)) if len(votes) > 1 else 1
        uncertainty = entropy / max_entropy if max_entropy > 0 else 0
    
    return final_select, "{:.2e}".format(uncertainty)


def bertseq(sequence, length, fillstr): 
	sequence_in = sequence.ljust(length, fillstr)
	emb = antiberty.embed([sequence_in], return_attention=False)
	embeddings_out = emb[0].cpu().detach()
	del emb
	return embeddings_out

#LABEL_TO_SPECIES = {0: "Camel", 1: "Human", 2: "Mouse", 3: "Rabbit", 4: "Rat", 5: "Rhesus"}
def specie_and_chain(seqs):
	print("** Calculating Specie and Chain [Fast]")
	if "_" in seqs:
		filled_seqs = antiberty.fill_masks([seqs])
		species_preds, chain_preds = antiberty.classify(filled_seqs)
	else:
		species_preds, chain_preds = antiberty.classify(seqs)
	return species_preds, chain_preds

# 最佳是AntiBERTy ResLSTM
def pred_Tm(seqs):
	Tm_pred, Tm_uncertainty = [], []
	for index, sequence in enumerate(tqdm(seqs, desc='** Calculating Tm:', dynamic_ncols=True)):
		if "_" in sequence:
			fill_seq = antiberty.fill_masks([sequence])
			filled_seq = ''.join(fill_seq)
			emb_in = bertseq(filled_seq, 200, "_").to(device)
			del fill_seq, filled_seq
		else:
			emb_in = bertseq(sequence, 200, "_").to(device)
		tmp_outs = [netsT[i](emb_in).cpu().detach().numpy() for i in range(5)]
		# Tm_value = sum(tmp_outs) / len(tmp_outs)
		tmp_outs_array = np.array(tmp_outs)
		Tm_value = np.mean(tmp_outs_array, axis=0)

		uncertainty = np.std(tmp_outs_array, axis=0)

		Tm_pred.append(round(Tm_value.item(), 2))
		Tm_uncertainty.append(round(uncertainty.item(), 2))
		del emb_in, tmp_outs, Tm_value
	return Tm_pred, Tm_uncertainty


# 最佳是AntiBERTy ResLSTM
def pred_Q(seqs):
	# class_dict = {0: "I", 1: "II", 2: "III", 3: "IV"}
	class_dict = {0: "1", 1: "2", 2: "3", 3: "4"}
	Q_pred, Q_entropy = [], []
	for index, sequence in enumerate(tqdm(seqs, desc='** Calculating Qclass:', dynamic_ncols=True)):
		if "_" in sequence:
			fill_seq = antiberty.fill_masks([sequence])
			filled_seq = ''.join(fill_seq)
			emb_in = bertseq(filled_seq, 200, "_").to(device)
			del fill_seq, filled_seq
		else:
			emb_in = bertseq(sequence, 200, "_").to(device)
		tmp_outs = [torch.argmax(torch.softmax(netsQ[i](emb_in), dim=-1)) for i in range(5)]
		# out = vote_select(tmp_outs)
		out, entropy = vote_select_with_uncertainty(tmp_outs)
		# Q_class = class_dict[out.item()]

		Q_pred.append(out)
		Q_entropy.append(entropy)
		del emb_in, tmp_outs, out
	return Q_pred, Q_entropy

# APP
def NBSTEM_app(fasta_input, output_name, seq_text):
	if fasta_input:
		names, seqs = read_fastatext(fasta_input)
	else:
		names = ["Nanobody"]
		seqs = [seq_text]
	print("== 5.Begin to predict: Tm, Qclass, Specie and Chain")
	specie_pred, chain_pred = specie_and_chain(seqs)
	Tm_pred, Tm_uncertainty = pred_Tm(seqs)
	Q_pred, Q_uncertainty = pred_Q(seqs)

	data = list(zip(names, Tm_pred, Tm_uncertainty, Q_pred, Q_uncertainty, specie_pred, seqs))
	df_seq = pd.DataFrame(data, columns=['ID', 'Tm', 'Tm_Uncertainty', 'Qclass', 'Q_Uncertainty', 'Specie', 'Sequence'])
	new_index = range(1, len(df_seq) + 1)
	df_seq = df_seq.set_index(pd.Index(new_index))
	del specie_pred, chain_pred, Tm_pred, Q_pred, data
	df_seq.to_csv(f"{output_name}")
	print(f"== 6.Finish ! The results are shown below or you can check file [{output_name}]")
	print("")
	print(df_seq)


if __name__ == '__main__':
	# args = parameter()
	# args = gui()
	NBSTEM_app(args.i, args.o, args.t)
