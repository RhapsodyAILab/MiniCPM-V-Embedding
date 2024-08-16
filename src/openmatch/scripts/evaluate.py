def eval_mrr(qrel, run, cutoff=None):
    """
    Compute MRR@cutoff manually.
    """
    mrr = 0.0
    num_ranked_q = 0
    results = {}
    for qid in qrel:
        if qid not in run:
            continue
        num_ranked_q += 1
        docid_and_score = [(docid, score) for docid, score in run[qid].items()]
        docid_and_score.sort(key=lambda x: x[1], reverse=True)
        for i, (docid, _) in enumerate(docid_and_score):
            rr = 0.0
            if cutoff is None or i < cutoff:
                if docid in qrel[qid] and qrel[qid][docid] > 0:
                    rr = 1.0 / (i + 1)
                    break
        results[qid] = rr
        mrr += rr
    mrr /= num_ranked_q
    results["all"] = mrr
    return results