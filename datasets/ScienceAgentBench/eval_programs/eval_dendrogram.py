from gpt4_visual_judge import encode_image, score_figure

def eval():
    pred_fpath = "pred_results/dendrogram.png"  
    gold_fpath = "benchmark/eval_programs/gold_results/dendrogram_gold.png"

    pred_fig = encode_image(pred_fpath)
    gold_fig = encode_image(gold_fpath)

    full_response, score = score_figure(pred_fig, gold_fig)

    return int(score >= 60), full_response

if __name__ == "__main__":
    print(eval())