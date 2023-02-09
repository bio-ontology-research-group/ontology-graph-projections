import pandas as pd
import sys


def analyze_result_metric(file_path, metric, criterion="max"):
    header = ["embed_dim", "margin", "reg", "batch_size", "lr", "mr", "mrr", "h1", "h10", "h100"]

    df = pd.read_csv(filename, header=None, names=header)

                                                        
    if criterion == "max":
        best = df.loc[df[metric].idxmax()].to_frame().T
    elif criterion == "min":
        best = df.loc[df[metric].idxmin()].to_frame().T
    best["mr"] = best["mr"].apply(lambda x: round(x, 2))
    best["mrr"] = best["mrr"].apply(lambda x: round(x, 2))
    best["h1"] = best["h1"].apply(lambda x: round(x*100, 2))
    best["h10"] = best["h10"].apply(lambda x: round(x*100, 2))
    best["h100"] = best["h100"].apply(lambda x: round(x*100, 2))
    best.round(4)
    return best
    
            
if __name__ == "__main__":
    
    filename = sys.argv[1]

    best_mr = analyze_result_metric(filename, "mr", criterion="min")
    best_mrr = analyze_result_metric(filename, "mrr")
    best_h1 = analyze_result_metric(filename, "h1")
    best_h10 = analyze_result_metric(filename, "h10")
    best_h100 = analyze_result_metric(filename, "h100")

    all_res = pd.concat([best_mr, best_mrr, best_h1, best_h10, best_h100])
    swap_list = ["embed_dim", "margin", "reg", "batch_size", "lr", "mrr", "mr", "h1", "h10", "h100"]
    all_res = all_res.reindex(columns=swap_list)
    print(all_res)

