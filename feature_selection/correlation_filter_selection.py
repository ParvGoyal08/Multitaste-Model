import os

import numpy as np
import pandas as pd

from common import EMBEDDINGS, load_embedding, parse_common_args, run_model_suite


def select_features_correlation(x_train, threshold=0.95, min_features=10):
    corr = np.corrcoef(x_train, rowvar=False)
    corr = np.nan_to_num(corr)
    keep = np.ones(corr.shape[0], dtype=bool)

    for i in range(corr.shape[0]):
        if not keep[i]:
            continue
        for j in range(i + 1, corr.shape[0]):
            if keep[j] and abs(corr[i, j]) >= threshold:
                keep[j] = False

    if keep.sum() < min_features:
        keep[:] = True
    return keep


def main():
    parser = parse_common_args("Correlation-filter selection + baseline model suite")
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--min-features", type=int, default=10)
    args = parser.parse_args()

    out_path = args.output or os.path.join(
        os.path.dirname(__file__),
        "correlation_filter_selection_results.csv",
    )

    all_rows = []
    for emb_name in EMBEDDINGS:
        x, y, feature_names = load_embedding(args.emb_dir, emb_name)
        mask = select_features_correlation(x, threshold=args.threshold, min_features=args.min_features)
        x_selected = x[:, mask]

        rows = run_model_suite(
            x_selected,
            y,
            embedding_name=emb_name,
            selected_feature_count=int(mask.sum()),
            seed=args.seed,
            test_size=args.test_size,
        )

        selected_names = [feature_names[i] for i, keep in enumerate(mask) if keep]
        for r in rows:
            r["Selector"] = "CorrelationFilter"
            r["Selected_Feature_Names"] = "|".join(selected_names)
        all_rows.extend(rows)

    pd.DataFrame(all_rows).to_csv(out_path, index=False)
    print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
