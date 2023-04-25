import argparse
import os, json
import numpy as np

def summarize_table_per_object_level(root_dir):

    l_datasets = None
    l_models = None
    l_op_points = None
    
    l_detectors = ["one_detector", "two_detectors", "baseline"]
    l_alias = ["SD", "MD", "B"]
    d_model = {"cascade_rcnn": "c-rcnn", "yolox": "yolox", "fcos": "fcos"}
    
    datasets = os.listdir(root_dir)
    for dataset in datasets:
        l_datasets = [dataset for dataset in datasets if os.path.isdir(os.path.join(root_dir, dataset))]
        if os.path.isdir(os.path.join(root_dir, dataset)):
            detectors = os.listdir(os.path.join(root_dir, dataset))
            for detector in detectors:  
                if os.path.isdir(os.path.join(root_dir, dataset, detector)):
                    b = []
                    op_points = os.listdir(os.path.join(root_dir, dataset, detector))
                    l_op_points = op_points
                    for op_point in op_points:
                        models = os.listdir(os.path.join(root_dir, dataset, detector, op_point))
                        l_models = models
                        break
                    break
                break
            break
        
    ######################################################################################################################
    # FALSE NEGATIVES
    ######################################################################################################################
    
    d_datasets_FN = {}
    for dataset in l_datasets:
        data = {}
        for model in l_models:
            data[model] = {l_detectors[0]: {}, l_detectors[1]: {}, l_detectors[2]: {}}
            for op_pt in l_op_points:
                
                for i in range(len(l_detectors)):
                    
                    current_dir = os.path.join(root_dir, dataset, l_detectors[i], op_pt, model)
                    img_FN = [filename for filename in os.listdir(current_dir) if filename.startswith("balance_FN")][0]
                    FN_max = img_FN.split(".jpg")[0].split("_")[-1]
                    img_FP = [filename for filename in os.listdir(current_dir) if filename.startswith("balance_FP")][0]
                    FP_max = img_FP.split(".jpg")[0].split("_")[-1]
                    
                    with open(os.path.join(os.path.join(current_dir, "per_object_experiment.json")), 'r') as f:
                        stats = json.load(f)
                        data[model][l_detectors[i]][op_pt] = {}
                        data[model][l_detectors[i]][op_pt] = stats[FN_max]
                        
        d_datasets_FN[dataset] = data
        
    d_datasets_FP = {}
    for dataset in l_datasets:
        data = {}
        for model in l_models:
            data[model] = {l_detectors[0]: {}, l_detectors[1]: {}, l_detectors[2]: {}}
            for op_pt in l_op_points:
                
                for i in range(len(l_detectors)):
                    
                    current_dir = os.path.join(root_dir, dataset, l_detectors[i], op_pt, model)
                    img_FN = [filename for filename in os.listdir(current_dir) if filename.startswith("balance_FN")][0]
                    FN_max = img_FN.split(".jpg")[0].split("_")[-1]
                    img_FP = [filename for filename in os.listdir(current_dir) if filename.startswith("balance_FP")][0]
                    FP_max = img_FP.split(".jpg")[0].split("_")[-1]
                    
                    with open(os.path.join(os.path.join(current_dir, "per_object_experiment.json")), 'r') as f:
                        stats = json.load(f)
                        data[model][l_detectors[i]][op_pt] = {}
                        data[model][l_detectors[i]][op_pt] = stats[FP_max]
                        
        d_datasets_FP[dataset] = data
        
    # create latex table
    with open(os.path.join(root_dir, "table_per_object_evaluation.txt"), 'w') as f:
        
        f.write("\\begin{table*}[!htb] \n")
        f.write("\caption{Offline experiment, operating point for person-detector: trade-off, $IoU=0.3$} \n")
        f.write("\label{tab:experiment:offline_experiment} \n")
        f.write("\centering \n")
        f.write("\\begin{tabular}{|c||c|c|c||c|c||c|c|}\n")
        f.write("\hline \n")
        f.write("\multirow{2}{*}{Method} & & & & \multicolumn{4}{|c|}{Per-object balances} \\\\ \n")
        f.write("  & TPs & FPs &  FNs & Detected FPs vs. undetected TPs & Balance & Detected FNs vs. ghost parts & Balance\\\\ \n")
        f.write("\hline \hline \n")
        
        f.write(" & & & & \multicolumn{4}{|c|}{\\textbf{COCO}} \\\\ \n")
        f.write("\hline \n")
            
        for model in d_datasets_FN["coco2014_full"].keys():
            
            tmp_1_fn = d_datasets_FN["coco2014_full"][model][l_detectors[0]]
            tmp_2_fn = d_datasets_FN["coco2014_full"][model][l_detectors[1]]
            tmp_3_fn = d_datasets_FN["coco2014_full"][model][l_detectors[2]]
            
            tmp_1_fp = d_datasets_FP["coco2014_full"][model][l_detectors[0]]
            tmp_2_fp = d_datasets_FP["coco2014_full"][model][l_detectors[1]]
            tmp_3_fp = d_datasets_FP["coco2014_full"][model][l_detectors[2]]
            
            f.write(f'{d_model[model]} ({l_alias[2]}) & {tmp_3_fn["tradeoff"]["tps"]} &  {tmp_3_fn["tradeoff"]["fps"]} &  {tmp_3_fn["tradeoff"]["fns"]} & {tmp_3_fp["tradeoff"]["detected_fps"]} / {tmp_3_fp["tradeoff"]["missed_tps"]} & {tmp_3_fp["tradeoff"]["detected_fps"] - tmp_3_fp["tradeoff"]["missed_tps"]} & {tmp_3_fn["tradeoff"]["detected_fns"]} / {tmp_3_fn["tradeoff"]["ghost_parts"]} & {tmp_3_fn["tradeoff"]["detected_fns"] - tmp_3_fn["tradeoff"]["ghost_parts"]} \\\\  \n') # & {tmp_3["o_pt_0.5"]["detected_fps"]} / {tmp_3["o_pt_0.5"]["n_tps"] - tmp_3["o_pt_0.5"]["detected_tps"]} & {tmp_3["o_pt_0.5"]["detected_fns"]} / {tmp_3["o_pt_0.5"]["n_ghost_parts"]} & {tmp_3["o_pt_0.8"]["detected_fps"]} / {tmp_3["o_pt_0.8"]["n_tps"] - tmp_3["o_pt_0.8"]["detected_tps"]} & {tmp_3["o_pt_0.8"]["detected_fns"]} / {tmp_3["o_pt_0.8"]["n_ghost_parts"]}  \\\\  \n')
            f.write(f'{d_model[model]} ({l_alias[1]}) & {tmp_2_fn["tradeoff"]["tps"]} &  {tmp_2_fn["tradeoff"]["fps"]} &  {tmp_2_fn["tradeoff"]["fns"]} & {tmp_2_fp["tradeoff"]["detected_fps"]} / {tmp_2_fp["tradeoff"]["missed_tps"]} & {tmp_2_fp["tradeoff"]["detected_fps"] - tmp_2_fp["tradeoff"]["missed_tps"]} & {tmp_2_fn["tradeoff"]["detected_fns"]} / {tmp_2_fn["tradeoff"]["ghost_parts"]} & {tmp_2_fn["tradeoff"]["detected_fns"] - tmp_2_fn["tradeoff"]["ghost_parts"]} \\\\  \n') # & {tmp_2["o_pt_0.5"]["detected_fps"]} / {tmp_2["o_pt_0.5"]["n_tps"] - tmp_2["o_pt_0.5"]["detected_tps"]} & {tmp_2["o_pt_0.5"]["detected_fns"]} / {tmp_2["o_pt_0.5"]["n_ghost_parts"]} & {tmp_2["o_pt_0.8"]["detected_fps"]} / {tmp_2["o_pt_0.8"]["n_tps"] - tmp_2["o_pt_0.8"]["detected_tps"]} & {tmp_2["o_pt_0.8"]["detected_fns"]} / {tmp_2["o_pt_0.8"]["n_ghost_parts"]}  \\\\  \n')
            f.write("\hline \n")
            f.write(f'{d_model[model]} ({l_alias[0]}) & {tmp_1_fn["tradeoff"]["tps"]} &  {tmp_1_fn["tradeoff"]["fps"]} &  {tmp_1_fn["tradeoff"]["fns"]} & {tmp_1_fp["tradeoff"]["detected_fps"]} / {tmp_1_fp["tradeoff"]["missed_tps"]} & {tmp_1_fp["tradeoff"]["detected_fps"] - tmp_1_fp["tradeoff"]["missed_tps"]} & {tmp_1_fn["tradeoff"]["detected_fns"]} / {tmp_1_fn["tradeoff"]["ghost_parts"]} & {tmp_1_fn["tradeoff"]["detected_fns"] - tmp_1_fn["tradeoff"]["ghost_parts"]} \\\\  \n') # & {tmp_1["o_pt_0.5"]["detected_fps"]} / {tmp_1["o_pt_0.5"]["n_tps"] - tmp_1["o_pt_0.5"]["detected_tps"]} & {tmp_1["o_pt_0.5"]["detected_fns"]} / {tmp_1["o_pt_0.5"]["n_ghost_parts"]} & {tmp_1["o_pt_0.8"]["detected_fps"]} / {tmp_1["o_pt_0.8"]["n_tps"] - tmp_1["o_pt_0.8"]["detected_tps"]} & {tmp_1["o_pt_0.8"]["detected_fns"]} / {tmp_1["o_pt_0.8"]["n_ghost_parts"]}  \\\\  \n')
            f.write("\hline \n")
            f.write("\hline \n")
              
        f.write("\hline \n")
        f.write(" & & & & \multicolumn{4}{|c|}{\\textbf{PascalVOC2010}} \\\\ \n")
        f.write("\hline \n")
            
        for model in d_datasets_FP["voc2010"].keys():
            
            tmp_1_fn = d_datasets_FN["voc2010"][model][l_detectors[0]]
            tmp_2_fn = d_datasets_FN["voc2010"][model][l_detectors[1]]
            tmp_3_fn = d_datasets_FN["voc2010"][model][l_detectors[2]]
            
            tmp_1_fp = d_datasets_FP["voc2010"][model][l_detectors[0]]
            tmp_2_fp = d_datasets_FP["voc2010"][model][l_detectors[1]]
            tmp_3_fp = d_datasets_FP["voc2010"][model][l_detectors[2]]
            
            f.write(f'{d_model[model]} ({l_alias[2]}) & {tmp_3_fn["tradeoff"]["tps"]} &  {tmp_3_fn["tradeoff"]["fps"]} &  {tmp_3_fn["tradeoff"]["fns"]} & {tmp_3_fp["tradeoff"]["detected_fps"]} / {tmp_3_fp["tradeoff"]["missed_tps"]} & {tmp_3_fp["tradeoff"]["detected_fps"] - tmp_3_fp["tradeoff"]["missed_tps"]} & {tmp_3_fn["tradeoff"]["detected_fns"]} / {tmp_3_fn["tradeoff"]["ghost_parts"]} & {tmp_3_fn["tradeoff"]["detected_fns"] - tmp_3_fn["tradeoff"]["ghost_parts"]} \\\\  \n') # & {tmp_3["o_pt_0.5"]["detected_fps"]} / {tmp_3["o_pt_0.5"]["n_tps"] - tmp_3["o_pt_0.5"]["detected_tps"]} & {tmp_3["o_pt_0.5"]["detected_fns"]} / {tmp_3["o_pt_0.5"]["n_ghost_parts"]} & {tmp_3["o_pt_0.8"]["detected_fps"]} / {tmp_3["o_pt_0.8"]["n_tps"] - tmp_3["o_pt_0.8"]["detected_tps"]} & {tmp_3["o_pt_0.8"]["detected_fns"]} / {tmp_3["o_pt_0.8"]["n_ghost_parts"]}  \\\\  \n')
            f.write(f'{d_model[model]} ({l_alias[1]}) & {tmp_2_fn["tradeoff"]["tps"]} &  {tmp_2_fn["tradeoff"]["fps"]} &  {tmp_2_fn["tradeoff"]["fns"]} & {tmp_2_fp["tradeoff"]["detected_fps"]} / {tmp_2_fp["tradeoff"]["missed_tps"]} & {tmp_2_fp["tradeoff"]["detected_fps"] - tmp_2_fp["tradeoff"]["missed_tps"]} & {tmp_2_fn["tradeoff"]["detected_fns"]} / {tmp_2_fn["tradeoff"]["ghost_parts"]} & {tmp_2_fn["tradeoff"]["detected_fns"] - tmp_2_fn["tradeoff"]["ghost_parts"]} \\\\  \n') # & {tmp_2["o_pt_0.5"]["detected_fps"]} / {tmp_2["o_pt_0.5"]["n_tps"] - tmp_2["o_pt_0.5"]["detected_tps"]} & {tmp_2["o_pt_0.5"]["detected_fns"]} / {tmp_2["o_pt_0.5"]["n_ghost_parts"]} & {tmp_2["o_pt_0.8"]["detected_fps"]} / {tmp_2["o_pt_0.8"]["n_tps"] - tmp_2["o_pt_0.8"]["detected_tps"]} & {tmp_2["o_pt_0.8"]["detected_fns"]} / {tmp_2["o_pt_0.8"]["n_ghost_parts"]}  \\\\  \n')
            f.write("\hline \n")
            f.write(f'{d_model[model]} ({l_alias[0]}) & {tmp_1_fn["tradeoff"]["tps"]} &  {tmp_1_fn["tradeoff"]["fps"]} &  {tmp_1_fn["tradeoff"]["fns"]} & {tmp_1_fp["tradeoff"]["detected_fps"]} / {tmp_1_fp["tradeoff"]["missed_tps"]} & {tmp_1_fp["tradeoff"]["detected_fps"] - tmp_1_fp["tradeoff"]["missed_tps"]} & {tmp_1_fn["tradeoff"]["detected_fns"]} / {tmp_1_fn["tradeoff"]["ghost_parts"]} & {tmp_1_fn["tradeoff"]["detected_fns"] - tmp_1_fn["tradeoff"]["ghost_parts"]} \\\\  \n') # & {tmp_1["o_pt_0.5"]["detected_fps"]} / {tmp_1["o_pt_0.5"]["n_tps"] - tmp_1["o_pt_0.5"]["detected_tps"]} & {tmp_1["o_pt_0.5"]["detected_fns"]} / {tmp_1["o_pt_0.5"]["n_ghost_parts"]} & {tmp_1["o_pt_0.8"]["detected_fps"]} / {tmp_1["o_pt_0.8"]["n_tps"] - tmp_1["o_pt_0.8"]["detected_tps"]} & {tmp_1["o_pt_0.8"]["detected_fns"]} / {tmp_1["o_pt_0.8"]["n_ghost_parts"]}  \\\\  \n')
            f.write("\hline \n")
            f.write("\hline \n")
                
        f.write("\end{tabular} \n")
        f.write("\\vspace{2pt} \n")
        f.write("\end{table*} \n")
        
def summarize_table_per_image_level(root_dir):

    l_datasets = None
    l_models = None
    l_op_points = None
    
    l_detectors = ["one_detector", "two_detectors", "baseline"]
    l_alias = ["SD", "MD", "B"]
    d_model = {"cascade_rcnn": "c-rcnn", "yolox": "yolox", "fcos": "fcos"}
    
    datasets = os.listdir(root_dir)
    for dataset in datasets:
        l_datasets = [dataset for dataset in datasets if os.path.isdir(os.path.join(root_dir, dataset))]
        if os.path.isdir(os.path.join(root_dir, dataset)):
            detectors = os.listdir(os.path.join(root_dir, dataset))
            for detector in detectors:  
                if os.path.isdir(os.path.join(root_dir, dataset, detector)):
                    b = []
                    op_points = os.listdir(os.path.join(root_dir, dataset, detector))
                    l_op_points = op_points
                    for op_point in op_points:
                        models = os.listdir(os.path.join(root_dir, dataset, detector, op_point))
                        l_models = models
                        break
                    break
                break
            break
        
    ######################################################################################################################
    # FALSE NEGATIVES
    ######################################################################################################################
    
    d_datasets = {}
    for dataset in l_datasets:
        data = {}
        for model in l_models:
            data[model] = {l_detectors[0]: {}, l_detectors[1]: {}, l_detectors[2]: {}}
            for op_pt in l_op_points:
                
                for i in range(len(l_detectors)):
                    
                    current_dir = os.path.join(root_dir, dataset, l_detectors[i], op_pt, model)
                    img_FN = [filename for filename in os.listdir(current_dir) if filename.startswith("analysis_FN")][0]
                    FN_max = img_FN.split(".jpg")[0].split("_")[-1]
                    img_FP = [filename for filename in os.listdir(current_dir) if filename.startswith("analysis_FP")][0]
                    FP_max = img_FP.split(".jpg")[0].split("_")[-1]
                                      
                    with open(os.path.join(os.path.join(current_dir, "per_image_experiment.json")), 'r') as f:
                        stats = json.load(f)

                        data[model][l_detectors[i]][op_pt] = {}
                        data[model][l_detectors[i]][op_pt]["nImages"] = stats["FN"][FN_max]["nImages"]
                        data[model][l_detectors[i]][op_pt]["N"] = stats["FN"][FN_max]["N"]
                        data[model][l_detectors[i]][op_pt]["tp"] = stats["FN"][FN_max]["tp"]
                        data[model][l_detectors[i]][op_pt]["fp"] = stats["FN"][FN_max]["fp"]
                        data[model][l_detectors[i]][op_pt]["precision"] = np.round(stats["FN"][FN_max]["precision"], decimals=2)
                        data[model][l_detectors[i]][op_pt]["recall"] = np.round(stats["FN"][FN_max]["recall"], decimals=2)
                        data[model][l_detectors[i]][op_pt]["MCC"] = np.round(stats["FN"][FN_max]["MCC"], decimals=2)
                        
        d_datasets[dataset] = data
    
    # create latex table
    with open(os.path.join(root_dir, "table_per_image_evaluation_FN.txt"), 'w') as f:
            f.write("\\begin{table*}[!htb] \n")
            f.write("\caption{Performance of the safety monitors in detecting whether at least one false negative detection is present in an image. As operation point, confidence scores that achieve maximal f1-score have been chosen for both, person and part detector. For computing TP, FP, and FN detection, an $IoU=0.3$ has been chosen.} \n")
            f.write("\label{tab:experiment:online_experiment_FN} \n")
            f.write("\centering \n")
            f.write("\\begin{tabular}{|c||c||c|c|c|c|c|}\n")
            f.write("\hline \n")
            f.write("\multirow{2}{*}{Method} & \multirow{2}{*}{Images with $\ge 1$ FN} & \multicolumn{5}{|c|}{Binary classification ($1$: $\ge 1$ FN / $0$: No FN)} \\\\ \n")
            f.write(" &  & TP & FP & precision & recall & MCC \\\\ \n")
            f.write("\hline \hline \n")
            
            f.write("\hline \n")
            f.write(" & \multicolumn{6}{|c|}{\\textbf{COCO}} \\\\ \n")
            f.write("\hline \n")
            
            for model in d_datasets["coco2014_full"].keys():
                
                tmp_1 = d_datasets["coco2014_full"][model][l_detectors[0]]
                tmp_2 = d_datasets["coco2014_full"][model][l_detectors[1]]
                tmp_3 = d_datasets["coco2014_full"][model][l_detectors[2]]
                
                f.write(f'{d_model[model]} ({l_alias[2]}) & {tmp_3["tradeoff"]["N"]} ({np.round(tmp_3["tradeoff"]["N"]/tmp_3["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_3["tradeoff"]["tp"]} &  {tmp_3["tradeoff"]["fp"]} & {tmp_3["tradeoff"]["precision"]} & {tmp_3["tradeoff"]["recall"]} & {tmp_3["tradeoff"]["MCC"]} \\\\  \n') #   & {tmp_3["o_pt_0.5"]["tp"]} & {tmp_3["o_pt_0.5"]["fp"]} & {tmp_3["o_pt_0.5"]["precision"]} & {tmp_3["o_pt_0.5"]["recall"]} & {tmp_3["o_pt_0.5"]["recall"]} & {tmp_3["o_pt_0.8"]["tp"]} & {tmp_3["o_pt_0.8"]["fp"]} & {tmp_3["o_pt_0.8"]["precision"]} & {tmp_3["o_pt_0.8"]["recall"]} & {tmp_3["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write(f'{d_model[model]} ({l_alias[1]}) & {tmp_2["tradeoff"]["N"]} ({np.round(tmp_2["tradeoff"]["N"]/tmp_2["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_2["tradeoff"]["tp"]} &  {tmp_2["tradeoff"]["fp"]} & {tmp_2["tradeoff"]["precision"]} & {tmp_2["tradeoff"]["recall"]} & {tmp_2["tradeoff"]["MCC"]} \\\\  \n') #   & {tmp_2["o_pt_0.5"]["tp"]} & {tmp_2["o_pt_0.5"]["fp"]} & {tmp_2["o_pt_0.5"]["precision"]} & {tmp_2["o_pt_0.5"]["recall"]} & {tmp_2["o_pt_0.5"]["recall"]} & {tmp_2["o_pt_0.8"]["tp"]} & {tmp_2["o_pt_0.8"]["fp"]} & {tmp_2["o_pt_0.8"]["precision"]} & {tmp_2["o_pt_0.8"]["recall"]} & {tmp_2["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write("\hline \n")
                f.write(f'{d_model[model]} ({l_alias[0]}) & {tmp_1["tradeoff"]["N"]} ({np.round(tmp_1["tradeoff"]["N"]/tmp_1["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_1["tradeoff"]["tp"]} &  {tmp_1["tradeoff"]["fp"]} & {tmp_1["tradeoff"]["precision"]} & {tmp_1["tradeoff"]["recall"]} & {tmp_1["tradeoff"]["MCC"]} \\\\  \n') #  & {tmp_1["o_pt_0.5"]["tp"]} & {tmp_1["o_pt_0.5"]["fp"]} & {tmp_1["o_pt_0.5"]["precision"]} & {tmp_1["o_pt_0.5"]["recall"]} & {tmp_1["o_pt_0.5"]["MCC"]} & {tmp_1["o_pt_0.8"]["tp"]} & {tmp_1["o_pt_0.8"]["fp"]} & {tmp_1["o_pt_0.8"]["precision"]} & {tmp_1["o_pt_0.8"]["recall"]} & {tmp_1["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write("\hline \n")
                f.write("\hline \n")
                
            f.write("\hline \n")
            f.write(" & \multicolumn{6}{|c|}{\\textbf{PascalVOC2010}} \\\\ \n")
            f.write("\hline \n")
            
            for model in d_datasets["voc2010"].keys():
                
                tmp_1 = d_datasets["voc2010"][model][l_detectors[0]]
                tmp_2 = d_datasets["voc2010"][model][l_detectors[1]]
                tmp_3 = d_datasets["voc2010"][model][l_detectors[2]]
                
                f.write(f'{d_model[model]} ({l_alias[2]}) & {tmp_3["tradeoff"]["N"]} ({np.round(tmp_3["tradeoff"]["N"]/tmp_3["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_3["tradeoff"]["tp"]} &  {tmp_3["tradeoff"]["fp"]} & {tmp_3["tradeoff"]["precision"]} & {tmp_3["tradeoff"]["recall"]} & {tmp_3["tradeoff"]["MCC"]} \\\\  \n') #   & {tmp_3["o_pt_0.5"]["tp"]} & {tmp_3["o_pt_0.5"]["fp"]} & {tmp_3["o_pt_0.5"]["precision"]} & {tmp_3["o_pt_0.5"]["recall"]} & {tmp_3["o_pt_0.5"]["recall"]} & {tmp_3["o_pt_0.8"]["tp"]} & {tmp_3["o_pt_0.8"]["fp"]} & {tmp_3["o_pt_0.8"]["precision"]} & {tmp_3["o_pt_0.8"]["recall"]} & {tmp_3["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write(f'{d_model[model]} ({l_alias[1]}) & {tmp_2["tradeoff"]["N"]} ({np.round(tmp_2["tradeoff"]["N"]/tmp_2["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_2["tradeoff"]["tp"]} &  {tmp_2["tradeoff"]["fp"]} & {tmp_2["tradeoff"]["precision"]} & {tmp_2["tradeoff"]["recall"]} & {tmp_2["tradeoff"]["MCC"]} \\\\  \n') #   & {tmp_2["o_pt_0.5"]["tp"]} & {tmp_2["o_pt_0.5"]["fp"]} & {tmp_2["o_pt_0.5"]["precision"]} & {tmp_2["o_pt_0.5"]["recall"]} & {tmp_2["o_pt_0.5"]["recall"]} & {tmp_2["o_pt_0.8"]["tp"]} & {tmp_2["o_pt_0.8"]["fp"]} & {tmp_2["o_pt_0.8"]["precision"]} & {tmp_2["o_pt_0.8"]["recall"]} & {tmp_2["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write("\hline \n")
                f.write(f'{d_model[model]} ({l_alias[0]}) & {tmp_1["tradeoff"]["N"]} ({np.round(tmp_1["tradeoff"]["N"]/tmp_1["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_1["tradeoff"]["tp"]} &  {tmp_1["tradeoff"]["fp"]} & {tmp_1["tradeoff"]["precision"]} & {tmp_1["tradeoff"]["recall"]} & {tmp_1["tradeoff"]["MCC"]} \\\\  \n') #  & {tmp_1["o_pt_0.5"]["tp"]} & {tmp_1["o_pt_0.5"]["fp"]} & {tmp_1["o_pt_0.5"]["precision"]} & {tmp_1["o_pt_0.5"]["recall"]} & {tmp_1["o_pt_0.5"]["MCC"]} & {tmp_1["o_pt_0.8"]["tp"]} & {tmp_1["o_pt_0.8"]["fp"]} & {tmp_1["o_pt_0.8"]["precision"]} & {tmp_1["o_pt_0.8"]["recall"]} & {tmp_1["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write("\hline \n")
                f.write("\hline \n")
                
            f.write("\end{tabular} \n")
            f.write("\\vspace{2pt} \n")
            f.write("\end{table*} \n")
            
            
    ######################################################################################################################
    # FALSE POSITIVES
    ######################################################################################################################
    
    d_datasets = {}
    for dataset in l_datasets:
        data = {}
        for model in l_models:
            data[model] = {l_detectors[0]: {}, l_detectors[1]: {}, l_detectors[2]: {}}
            for op_pt in l_op_points:
                
                for i in range(len(l_detectors)):
                    
                    current_dir = os.path.join(root_dir, dataset, l_detectors[i], op_pt, model)
                    img_FN = [filename for filename in os.listdir(current_dir) if filename.startswith("analysis_FN")][0]
                    FN_max = img_FN.split(".jpg")[0].split("_")[-1]
                    img_FP = [filename for filename in os.listdir(current_dir) if filename.startswith("analysis_FP")][0]
                    FP_max = img_FP.split(".jpg")[0].split("_")[-1]
                                      
                    with open(os.path.join(os.path.join(current_dir, "per_image_experiment.json")), 'r') as f:
                        stats = json.load(f)
                        
                        data[model][l_detectors[i]][op_pt] = {}
                        data[model][l_detectors[i]][op_pt]["nImages"] = stats["FP"][FP_max]["nImages"]
                        data[model][l_detectors[i]][op_pt]["N"] = stats["FP"][FP_max]["N"]
                        data[model][l_detectors[i]][op_pt]["tp"] = stats["FP"][FP_max]["tp"]
                        data[model][l_detectors[i]][op_pt]["fp"] = stats["FP"][FP_max]["fp"]
                        data[model][l_detectors[i]][op_pt]["precision"] = np.round(stats["FP"][FP_max]["precision"], decimals=2)
                        data[model][l_detectors[i]][op_pt]["recall"] = np.round(stats["FP"][FP_max]["recall"], decimals=2)
                        data[model][l_detectors[i]][op_pt]["MCC"] = np.round(stats["FP"][FP_max]["MCC"], decimals=2)
                        
        d_datasets[dataset] = data
    
    # create latex table
    with open(os.path.join(root_dir, "table_per_image_evaluation_FP.txt"), 'w') as f:
            f.write("\\begin{table*}[!htb] \n")
            f.write("\caption{Performance of the safety monitors in detecting whether at least one false positive detection is present in an image. As operation point, confidence scores that achieve maximal f1-score have been chosen for both, person and part detector. For computing TP, FP, and FN detection, an $IoU=0.3$ has been chosen.} \n")
            f.write("\label{tab:experiment:online_experiment_FN} \n")
            f.write("\centering \n")
            f.write("\\begin{tabular}{|c||c||c|c|c|c|c|}\n")
            f.write("\hline \n")
            f.write("\multirow{2}{*}{Method} & \multirow{2}{*}{Images with $\ge 1$ FP} & \multicolumn{5}{|c|}{Binary classification ($1$: $\ge 1$ FP / $0$: No FP)} \\\\ \n")
            f.write(" &  & TP & FP & precision & recall & MCC \\\\ \n")
            f.write("\hline \hline \n")
            
            f.write("\hline \n")
            f.write(" & \multicolumn{6}{|c|}{\\textbf{COCO}} \\\\ \n")
            f.write("\hline \n")
            
            for model in d_datasets["coco2014_full"].keys():
                
                tmp_1 = d_datasets["coco2014_full"][model][l_detectors[0]]
                tmp_2 = d_datasets["coco2014_full"][model][l_detectors[1]]
                tmp_3 = d_datasets["coco2014_full"][model][l_detectors[2]]
                
                f.write(f'{d_model[model]} ({l_alias[2]}) & {tmp_3["tradeoff"]["N"]} ({np.round(tmp_3["tradeoff"]["N"]/tmp_3["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_3["tradeoff"]["tp"]} &  {tmp_3["tradeoff"]["fp"]} & {tmp_3["tradeoff"]["precision"]} & {tmp_3["tradeoff"]["recall"]} & {tmp_3["tradeoff"]["MCC"]} \\\\  \n') #   & {tmp_3["o_pt_0.5"]["tp"]} & {tmp_3["o_pt_0.5"]["fp"]} & {tmp_3["o_pt_0.5"]["precision"]} & {tmp_3["o_pt_0.5"]["recall"]} & {tmp_3["o_pt_0.5"]["recall"]} & {tmp_3["o_pt_0.8"]["tp"]} & {tmp_3["o_pt_0.8"]["fp"]} & {tmp_3["o_pt_0.8"]["precision"]} & {tmp_3["o_pt_0.8"]["recall"]} & {tmp_3["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write(f'{d_model[model]} ({l_alias[1]}) & {tmp_2["tradeoff"]["N"]} ({np.round(tmp_2["tradeoff"]["N"]/tmp_2["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_2["tradeoff"]["tp"]} &  {tmp_2["tradeoff"]["fp"]} & {tmp_2["tradeoff"]["precision"]} & {tmp_2["tradeoff"]["recall"]} & {tmp_2["tradeoff"]["MCC"]} \\\\  \n') #   & {tmp_2["o_pt_0.5"]["tp"]} & {tmp_2["o_pt_0.5"]["fp"]} & {tmp_2["o_pt_0.5"]["precision"]} & {tmp_2["o_pt_0.5"]["recall"]} & {tmp_2["o_pt_0.5"]["recall"]} & {tmp_2["o_pt_0.8"]["tp"]} & {tmp_2["o_pt_0.8"]["fp"]} & {tmp_2["o_pt_0.8"]["precision"]} & {tmp_2["o_pt_0.8"]["recall"]} & {tmp_2["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write("\hline \n")
                f.write(f'{d_model[model]} ({l_alias[0]}) & {tmp_1["tradeoff"]["N"]} ({np.round(tmp_1["tradeoff"]["N"]/tmp_1["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_1["tradeoff"]["tp"]} &  {tmp_1["tradeoff"]["fp"]} & {tmp_1["tradeoff"]["precision"]} & {tmp_1["tradeoff"]["recall"]} & {tmp_1["tradeoff"]["MCC"]} \\\\  \n') #  & {tmp_1["o_pt_0.5"]["tp"]} & {tmp_1["o_pt_0.5"]["fp"]} & {tmp_1["o_pt_0.5"]["precision"]} & {tmp_1["o_pt_0.5"]["recall"]} & {tmp_1["o_pt_0.5"]["MCC"]} & {tmp_1["o_pt_0.8"]["tp"]} & {tmp_1["o_pt_0.8"]["fp"]} & {tmp_1["o_pt_0.8"]["precision"]} & {tmp_1["o_pt_0.8"]["recall"]} & {tmp_1["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write("\hline \n")
                f.write("\hline \n")
                
            f.write("\hline \n")
            f.write(" & \multicolumn{6}{|c|}{\\textbf{PascalVOC2010}} \\\\ \n")
            f.write("\hline \n")
            
            for model in d_datasets["voc2010"].keys():
                
                tmp_1 = d_datasets["voc2010"][model][l_detectors[0]]
                tmp_2 = d_datasets["voc2010"][model][l_detectors[1]]
                tmp_3 = d_datasets["voc2010"][model][l_detectors[2]]
                
                f.write(f'{d_model[model]} ({l_alias[2]}) & {tmp_3["tradeoff"]["N"]} ({np.round(tmp_3["tradeoff"]["N"]/tmp_3["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_3["tradeoff"]["tp"]} &  {tmp_3["tradeoff"]["fp"]} & {tmp_3["tradeoff"]["precision"]} & {tmp_3["tradeoff"]["recall"]} & {tmp_3["tradeoff"]["MCC"]} \\\\  \n') #   & {tmp_3["o_pt_0.5"]["tp"]} & {tmp_3["o_pt_0.5"]["fp"]} & {tmp_3["o_pt_0.5"]["precision"]} & {tmp_3["o_pt_0.5"]["recall"]} & {tmp_3["o_pt_0.5"]["recall"]} & {tmp_3["o_pt_0.8"]["tp"]} & {tmp_3["o_pt_0.8"]["fp"]} & {tmp_3["o_pt_0.8"]["precision"]} & {tmp_3["o_pt_0.8"]["recall"]} & {tmp_3["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write(f'{d_model[model]} ({l_alias[1]}) & {tmp_2["tradeoff"]["N"]} ({np.round(tmp_2["tradeoff"]["N"]/tmp_2["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_2["tradeoff"]["tp"]} &  {tmp_2["tradeoff"]["fp"]} & {tmp_2["tradeoff"]["precision"]} & {tmp_2["tradeoff"]["recall"]} & {tmp_2["tradeoff"]["MCC"]} \\\\  \n') #   & {tmp_2["o_pt_0.5"]["tp"]} & {tmp_2["o_pt_0.5"]["fp"]} & {tmp_2["o_pt_0.5"]["precision"]} & {tmp_2["o_pt_0.5"]["recall"]} & {tmp_2["o_pt_0.5"]["recall"]} & {tmp_2["o_pt_0.8"]["tp"]} & {tmp_2["o_pt_0.8"]["fp"]} & {tmp_2["o_pt_0.8"]["precision"]} & {tmp_2["o_pt_0.8"]["recall"]} & {tmp_2["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write("\hline \n")
                f.write(f'{d_model[model]} ({l_alias[0]}) & {tmp_1["tradeoff"]["N"]} ({np.round(tmp_1["tradeoff"]["N"]/tmp_1["tradeoff"]["nImages"]*100, decimals=1)}\%) &  {tmp_1["tradeoff"]["tp"]} &  {tmp_1["tradeoff"]["fp"]} & {tmp_1["tradeoff"]["precision"]} & {tmp_1["tradeoff"]["recall"]} & {tmp_1["tradeoff"]["MCC"]} \\\\  \n') #  & {tmp_1["o_pt_0.5"]["tp"]} & {tmp_1["o_pt_0.5"]["fp"]} & {tmp_1["o_pt_0.5"]["precision"]} & {tmp_1["o_pt_0.5"]["recall"]} & {tmp_1["o_pt_0.5"]["MCC"]} & {tmp_1["o_pt_0.8"]["tp"]} & {tmp_1["o_pt_0.8"]["fp"]} & {tmp_1["o_pt_0.8"]["precision"]} & {tmp_1["o_pt_0.8"]["recall"]} & {tmp_1["o_pt_0.8"]["MCC"]} \\\\  \n')
                f.write("\hline \n")
                f.write("\hline \n")
                
            f.write("\end{tabular} \n")
            f.write("\\vspace{2pt} \n")
            f.write("\end{table*} \n")
                
if __name__ == "__main__":
    
    args = argparse.ArgumentParser(description='Model Characteristics')

    args.add_argument('--root', type=str, help="path to results folder")
    args = args.parse_args()
    
    summarize_table_per_image_level(root_dir=args.root)
    summarize_table_per_object_level(root_dir=args.root)
