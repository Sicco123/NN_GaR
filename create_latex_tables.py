import numpy as np
import pandas as pd
import pickle

date = '221014'
M = 500
steps_ahead = [1, 2, 3, 4, 5, 6]
quantile_levels = np.array([0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
test_stats = ['Coverage', 'Tick Loss']
experiment_names = ['Linear Regression', 'Sequence-to-Sequence']

with open(f'simulation_results/{date}_{M}_stats.pkl', "rb") as f:
    results = pickle.load(f)

K, L, M, N = results.shape


cols = ['1','2','3','4','5','6']
rows = [str(q) for q in quantile_levels]

for k in range(K):
    for n in range(N):
        sub_result = results[k,:,:,n]
        df_res = pd.DataFrame(sub_result, columns= cols, index=rows)
        s = df_res.style.format(precision = 3)#float_format="{:0.3f}".format)
        latex_tab = s.to_latex(hrules = True,  caption = f'{experiment_names[k]} - {test_stats[n]}',
                                    label=f'tab: {experiment_names[k]} {test_stats[n][:3]}', position = 'H!')
        #latex_tab = df_res.to_latex(hrules = True, float_format="{:0.3f}".format, caption = f'{experiment_names[k]} - {test_stats[n]}',
        #                           label=f'tab: {experiment_names[k]} {test_stats[n][:3]}', position = 'H!')

        if k == 0 and n == 0:
            write_append = 'wt'

            with open(f'simulation_results/{date}_{M}_latex.tex', f'{write_append}') as file:
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
                file.write("\\usepackage{booktabs}")
                file.write(latex_tab)
                file.write('\n')

