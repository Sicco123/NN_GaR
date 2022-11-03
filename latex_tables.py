import numpy as np
import pandas as pd
import pickle
#
# date = '221014'
# M = 500
# steps_ahead = [1, 2, 3, 4, 5, 6]
# quantile_levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
# test_stats = ['Coverage', 'Tick Loss']
# experiment_names = ['Linear Regression', 'Sequence-to-Sequence']
#



def make_latex_table( steps_ahead, quantile_levels, test_stats, experiment_names, M, date):
    with open(f'simulation_results/{date}_{M}_stats.pkl', "rb") as f:
        results = pickle.load(f)

    K, L, S, N = results.shape

    cols = [str(p) for p in steps_ahead]
    rows = [str(q) for q in quantile_levels]

    for k in range(K):
        for n in range(N):
            sub_result = results[k,:,:,n]
            df_res = pd.DataFrame(sub_result, columns= cols, index=rows)
            s = df_res.style.format(precision = 3)
            latex_tab = s.to_latex(hrules = True, position_float = "centering",  caption = f'{experiment_names[k]} - {test_stats[n]}',
                                        label=f'tab: {experiment_names[k]} {test_stats[n][:3]}', position = 'h!')

            if k == 0 and n == 0:
                write_append = 'wt'

                with open(f'simulation_results/{date}_{M}_latex.tex', f'{write_append}') as file:

                    file.write("\\documentclass{article}\n")
                    file.write("\\usepackage[utf8]{inputenc}")
                    file.write("\\usepackage{booktabs} \n")
                    file.write("\\begin{document}\n")
                    file.write(latex_tab)
                    file.write('\n')
            elif k == K-1 and n == N-1:
                write_append = 'at'
                with open(f'simulation_results/{date}_{M}_latex.tex', f'{write_append}') as file:
                    file.write(latex_tab)
                    file.write('\n')
                    file.write('\\end{document}')
            else:
                write_append = 'at'
                with open(f'simulation_results/{date}_{M}_latex.tex', f'{write_append}') as file:
                    file.write(latex_tab)
                    file.write('\n')

