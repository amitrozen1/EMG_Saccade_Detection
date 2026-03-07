#!/bin/env python3.9

import argparse
from os.path import dirname, join

from matplotlib import pyplot as plt
from matplotlib import use as use_backend
from pandas import DataFrame, read_pickle
from pyperclip import copy
from tqdm import tqdm

from shared import METHODS
from shared import DFLine, Metric, Status, iterate_matlab_folder, read_matlab

DATA_PATH = join(dirname(dirname(__file__)), 'data')
ANGLES = [20, 30, 60]


def extract_mse_dataframe():
    lines = []

    pbar = tqdm(iterate_matlab_folder(DATA_PATH))
    for record in pbar:
        pbar.set_description(f'Extracting MSE from "{record.filename}"')
        downsampled = record.downsampled(5)
        for line in downsampled.mse_lines():
            lines.append(line.df_row)

    df = DataFrame(
        lines,
        columns=DFLine.columns(Metric.MSE)
    )

    df.to_pickle(join(DATA_PATH, 'mse.pkl.xz'), compression='infer')


def extract_biomarkers_dataframes():
    peak_velocity_lines = []
    duration_lines = []
    latency_lines = []

    pbar = tqdm(iterate_matlab_folder(DATA_PATH))
    for record in pbar:
        pbar.set_description(f'Extracting biomarkers from "{record.filename}"')
        downsampled = record.downsampled(5)
        for line in downsampled.peak_velocity_lines():
            peak_velocity_lines.append(line.df_row)

        for line in downsampled.time_lines():
            if line.metric == Metric.Latency:
                latency_lines.append(line.df_row)
            elif line.metric == Metric.Duration:
                duration_lines.append(line.df_row)

    peak_velocity_df = DataFrame(
        peak_velocity_lines,
        columns=DFLine.columns(Metric.PeakVelocity)
    )

    peak_velocity_df.to_pickle(join(DATA_PATH, 'peak_velocities.pkl.xz'), compression='infer')

    latency_df = DataFrame(
        latency_lines,
        columns=DFLine.columns(Metric.Latency)
    )

    latency_df.to_pickle(join(DATA_PATH, 'latencies.pkl.xz'), compression='infer')

    durations_df = DataFrame(
        duration_lines,
        columns=DFLine.columns(Metric.Duration)
    )

    durations_df.to_pickle(join(DATA_PATH, 'durations.pkl.xz'), compression='infer')


def describe_data():
    saccades = []

    data = {
        status: {
            angle: 0
            for angle in ANGLES
        }
        for status in Status
    }

    pbar = tqdm(iterate_matlab_folder(DATA_PATH))
    for record in pbar:
        pbar.set_description(f'Processing {record.filename}')
        data[record.status][record.angle] += 1
        saccades.append(record.saccades_count)

    params = []
    total = 0
    for status in Status:
        status_total = 0
        for angle in ANGLES:
            current = data[status][angle]
            params.append(current)
            status_total += current
        params.append(status_total)
        total += status_total

    for angle in ANGLES:
        angle_total = 0
        for status in Status:
            angle_total += data[status][angle]
        params.append(angle_total)

    params.append(total)

    table = '''
\\begin{table}[h]
    \\centering
    \\begin{tabular}{rccccr}
        \\toprule
        \\textbf{Class} & $\\mathbf{20^\\circ}$ & $\\mathbf{30^\\circ}$ & $\\mathbf{60^\\circ}$ &
        \\textbf{Total}\\\\
        \\midrule
        \\textit{Healthy} & %d & %d & %d & %d\\\\
        \\textit{SCA2-Sick} & %d & %d & %d & %d\\\\
        \\midrule
        \\textit{Total} & %d & %d & %d & %d\\\\
        \\bottomrule
    \\end{tabular}
    \\caption{Records distribution\\label{tbl:records}}
\\end{table}
    ''' % tuple(params)

    copy(table)

    print(table)
    print()
    print(f'Saccades Count: {sum(saccades)}')


def exact_saccades_stats():
    saccades = []
    pbar = tqdm(iterate_matlab_folder(DATA_PATH))
    for record in pbar:
        pbar.set_description(f'Processing {record.filename}')

        downsampled = record.downsampled(5)

        for onset, offset in downsampled.saccades(downsampled.V0):
            saccades.append([
                downsampled.status.value,
                downsampled.angle,
                downsampled.noise,
                (offset - onset) * downsampled.h,
                max(abs(downsampled.V0[onset:offset]))
            ])


    saccades_df = DataFrame(
        saccades,
        columns=['Status', 'Angle', 'Noise', 'Duration', 'PeakVelocity']
    )

    saccades_df.to_pickle(join(DATA_PATH, 'exact_saccades.pkl.xz'), compression='infer')

    print('Job completed')


def figure_3cd_vs_5cd():
    use_backend('Qt5Agg')

    records = list(read_matlab('../data/RegScSimul20_1000_allNoisesDC_0.1_Enfermo.mat'))

    rec = records[0].downsampled(5)

    CD3 = rec.velocities('cd3')
    SNR5 = rec.velocities('snr5')

    plt.rcParams['figure.figsize'] = (8, 6)

    plt.subplot(3, 1, 1)
    plt.plot(rec.X, rec.Y)
    plt.title('Movement signal')
    plt.xlabel('Seconds (s)')
    plt.ylabel('Angle ($\circ$)')

    plt.subplot(3, 1, 2)
    plt.plot(rec.X, CD3)
    plt.title('Differentiated with 3 points central difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity ($\circ/s$)')

    plt.subplot(3, 1, 3)
    plt.plot(rec.X, SNR5)
    plt.title('Differentiated with 5 points central difference')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity ($\circ/s$)')

    plt.tight_layout()
    plt.savefig('../article/figures/3cd_vs_5cd.eps', format='eps')

    plt.show()


def figure_cd_vs_sl():
    use_backend('Qt5Agg')

    records = list(read_matlab('../data/RegScSimul20_1000_allNoisesDC_0.5_Sano.mat'))

    rec = records[0].downsampled(5)


    FROM_SAMPLE = 300
    TO_SAMPLE = 500
    X = rec.X[FROM_SAMPLE:TO_SAMPLE]
    V0 = rec.V0[FROM_SAMPLE:TO_SAMPLE]
    CD3 = rec.velocities('cd3')[FROM_SAMPLE:TO_SAMPLE]
    SL7 = rec.velocities('sl7')[FROM_SAMPLE:TO_SAMPLE]

    plt.rcParams['figure.figsize'] = (8, 6)

    plt.subplot(2, 1, 1)
    plt.plot(X, CD3, label='CD3 output')
    plt.plot(X, V0, label='Synthetic velocity')
    plt.title('Central Difference by 3 points')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity ($\circ/s$)')
    plt.ylim([-50, 350])
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(X, SL7, label='SL7 output')
    plt.plot(X, V0, label='Synthetic velocity')
    plt.title('Super Lanczos by 5 points')
    plt.xlabel('Time (s)')
    plt.ylabel('Velocity ($\circ/s$)')
    plt.ylim([-50, 350])
    plt.legend()

    plt.tight_layout()
    plt.savefig('../article/figures/cd_vs_sl.eps', format='eps')

    plt.show()


def detected_saccades_analysis():
    df_lines = []

    stats = {
        method: {
            'unidentified': 0,
            'overidentified': 0,
        }
        for method in METHODS.keys()
        if method not in {'cd3', 'cd5', 'cd7', 'cd9'}
    }

    pbar = tqdm(iterate_matlab_folder(DATA_PATH))
    for record in pbar:
        pbar.set_description(f'Processing {record.filename}')
        downsampled = record.downsampled(5)

        for line in downsampled.detected_saccades_lines():
            df_lines.append(line.df_row)

            if line.value < 0:
                stats[line.method]['unidentified'] += int(line.value)
            elif line.value > 0:
                stats[line.method]['overidentified'] += int(line.value)

    df = DataFrame(
        df_lines,
        columns=DFLine.columns(Metric.DetectedSaccades)
    )

    filename = 'detected_saccades.pkl.xz'
    df.to_pickle(
        join(DATA_PATH, filename),
        compression='infer'
    )

    print(f'Filename: "{filename}" generated')

    records = list(read_matlab('../data/RegScSimul20_1000_allNoisesDC_0.5_Enfermo.mat'))
    rec = records[0].downsampled(5)

    labels = []
    unidentified = []
    overidentified = []
    for method, m_stats in sorted(
        stats.items(),
        key=lambda x: abs(x[1]['unidentified']) + x[1]['overidentified']
    ):
        labels.append(method)
        unidentified.append(m_stats['unidentified'])
        overidentified.append(m_stats['overidentified'])

    plt.bar(labels, unidentified, label='Unidentified saccades')
    plt.bar(labels, overidentified, label='Overidentified saccades')
    plt.xlabel('Method')
    plt.ylabel('Amount of saccades')
    plt.legend()
    plt.title('Missidentified saccades')
    plt.tight_layout()

    plt.savefig('../article/figures/identified_saccades_errors.eps', format='eps')

    plt.show()


def biomarkers_boxplot():
    METHODS = ['l5', 'l7', 'l9', 'l11', 'l13', 'sl7', 'sl9', 'sl11', 'snr5', 'snr7', 'snr9', 'snr11']

    peak_velocities_df = read_pickle('../data/peak_velocities.pkl.xz')
    latencies_df = read_pickle('../data/latencies.pkl.xz')
    duration_df = read_pickle('../data/durations.pkl.xz')

    plt.rcParams['figure.figsize'] = (6, 10)

    peak_velocities, latencies, durations = [], [], []
    for method in METHODS:
        peak_velocities.append(peak_velocities_df[peak_velocities_df.Method == method]['PeakVelocity'].values)
        latencies.append(latencies_df[latencies_df.Method == method]['Latency'].values)
        durations.append(duration_df[duration_df.Method == method]['Duration'].values)


    plt.subplot(3, 1, 1)
    bp = plt.boxplot(peak_velocities, labels=METHODS, notch=True)
    plt.title('Peak Velocity')
    plt.ylabel('Error')
    plt.setp(bp['boxes'][7], color='blue')
    plt.setp(bp['fliers'][7], markeredgecolor='blue')

    plt.subplot(3, 1, 2)
    bp = plt.boxplot(latencies, labels=METHODS, notch=True)
    plt.title('Latency')
    plt.ylabel('Error')
    plt.setp(bp['boxes'][7], color='blue')
    plt.setp(bp['fliers'][7], markeredgecolor='blue')
    plt.setp(bp['boxes'][10], color='blue')
    plt.setp(bp['fliers'][10], markeredgecolor='blue')
    plt.setp(bp['boxes'][11], color='blue')
    plt.setp(bp['fliers'][11], markeredgecolor='blue')

    plt.subplot(3, 1, 3)
    bp = plt.boxplot(durations, labels=METHODS, notch=True)
    plt.title('Duration')
    plt.xlabel('Method')
    plt.ylabel('Error')
    plt.setp(bp['boxes'][2], color='blue')
    plt.setp(bp['fliers'][2], markeredgecolor='blue')
    plt.setp(bp['boxes'][3], color='blue')
    plt.setp(bp['fliers'][3], markeredgecolor='blue')

    plt.tight_layout()
    plt.savefig('../article/figures/biomarkers_boxplot.eps', format='eps')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='DiffExp',
        description='Differentiation Methods Selection Experiment'
    )
    parser.add_argument(
        '-mse --extract-mse-df',
        action='store_true',
        dest='extract_mse_dataframe',
        help='Extract MSE DataFrame'
    )

    parser.add_argument(
        '-bmk --extract-biomarkers-dataframes',
        action='store_true',
        dest='extract_biomarkers_dataframes',
        help='Generate all data frames'
    )
    parser.add_argument(
        '-dd --describe-data',
        action='store_true',
        dest='describe_data',
        help='Show data distribution stats'
    )
    parser.add_argument(
        '-ess --exact-saccades-stats',
        action='store_true',
        dest='exact_saccades_stats',
        help='Generate DataFrame with saccades reference stats'
    )

    parser.add_argument(
        '-f3v5cd --figure-3cd-vs-5cd',
        action='store_true',
        dest='figure_3cd_vs_5cd',
        help='Generate a figure showing the effects of central difference differentiation'
    )

    parser.add_argument(
        '-fcvs --figure-cd-vs-sl',
        action='store_true',
        dest='figure_cd_vs_sl',
        help='Generate a figure showing central difference vs super lanczos'
    )

    parser.add_argument(
        '-dsa --detected-saccades-analysis',
        action='store_true',
        dest='detected_saccades_analysis',
        help='Analyze detected saccades and make a bar plot'
    )

    parser.add_argument(
        '-bbp --biomarkers-box-plot',
        action='store_true',
        dest='biomarkers_boxplot',
        help='Show biomarkers calculation errors boxplot'
    )

    args = parser.parse_args()

    option_count = 0

    if args.extract_mse_dataframe:
        extract_mse_dataframe()
        option_count += 1

    if args.extract_biomarkers_dataframes:
        extract_biomarkers_dataframes()
        option_count += 1

    if args.describe_data:
        describe_data()
        option_count += 1

    if args.exact_saccades_stats:
        exact_saccades_stats()
        option_count += 1

    if args.figure_3cd_vs_5cd:
        figure_3cd_vs_5cd()
        option_count += 1

    if args.figure_cd_vs_sl:
        figure_cd_vs_sl()
        option_count += 1

    if args.detected_saccades_analysis:
        detected_saccades_analysis()
        option_count += 1

    if args.biomarkers_boxplot:
        biomarkers_boxplot()
        option_count += 1

    if option_count == 0:
        parser.print_help()
