import pandas as pd
import click as ck
@ck.command()
@ck.option('--file_to_filter_on', '-f1', type=ck.Path(exists=True), required=True)
@ck.option('--file_to_filter_with', '-f2', type=ck.Path(exists=True), required=True)
def main(file_to_filter_on, file_to_filter_with):
    df1 = pd.read_csv(file_to_filter_on, sep=',', header=None)
    df2 = pd.read_csv(file_to_filter_with, sep=',', header=None)
    diff = set(df1.index).difference(set(df2.index))
    df3 = df1.loc[list(diff)].dropna()
    output_file = file_to_filter_on.replace('.csv', '_filtered.csv')
    df3.to_csv(output_file, index=False, header=False)
    print(f'Filtered file saved to {output_file}')

if __name__ == '__main__':
    main()
