import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Reward plot
rtlf_model_logdir = "./logs/train_ddpg_uneven_teacher_gamma_0.1/2025_02_27_03_10_21"
# phydrl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_0.4/2025_02_27_13_13_32"
phydrl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_0.45/2025_02_28_12_12_49"

# drl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_0.8/2025_02_27_19_07_51"
drl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_1/2025_02_28_01_31_47"
#
# Falls plot
# rtlf_model_logdir = "./logs/train_ddpg_uneven_teacher_gamma_0.1/2025_02_25_12_18_17"
# phydrl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_0.4/2025_02_27_13_13_32"
# drl_model_logdir = "./logs/train_ddpg_uneven_student_gamma_1/2025_02_26_17_58_41"


def load_event_file(log_dir):
    """Load the event file in the log dir"""
    event_files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if "tfevents" in f]
    event_file = sorted(event_files, key=os.path.getmtime)[-1]

    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    return event_acc


def plot_for_event(event_acc, tag, color, label, plot_distribution=True):
    if tag in event_acc.Tags()['scalars']:
        events = event_acc.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])

        # plot with distribution
        if plot_distribution:
            # Compute the moving average and standard deviation.
            window_size = 120  # window size
            rewards_mean = np.convolve(values, np.ones(window_size) / window_size, mode='valid')
            rewards_std = np.array([np.std(values[max(0, i - window_size):i]) for i in range(len(values))])

            plt.plot(steps[:len(rewards_mean)], rewards_mean, label=label, color=color, linewidth=5)
            plt.fill_between(steps[:len(rewards_mean)],
                             rewards_mean - rewards_std[:len(rewards_mean)],
                             rewards_mean + rewards_std[:len(rewards_mean)],
                             color=color, alpha=0.2)  # range of standard
        else:
            plt.plot(steps, values, label=label, color=color, linewidth=5)


def summary_fig_plot(tag="Train/mean_reward", plot_distribution=True):
    trlf_event = load_event_file(rtlf_model_logdir)
    phydrl_event = load_event_file(phydrl_model_logdir)
    drl_event = load_event_file(drl_model_logdir)

    # tag = "Train/mean_reward"

    # Figure plot
    fig = plt.figure(figsize=(13, 12))
    plot_for_event(trlf_event, tag, color='green', label="Ours", plot_distribution=plot_distribution)
    plot_for_event(phydrl_event, tag, color='blue', label="Phy-DRL", plot_distribution=plot_distribution)
    plot_for_event(drl_event, tag, color='red', label="CLF-DRL", plot_distribution=plot_distribution)

    plt.xlabel("Training Episode", fontsize=30)
    plt.ylabel("Times of Falls", fontsize=30)
    # plt.title("Training Reward with Variance")
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.legend(loc="lower right", fontsize=25)
    # plt.grid()
    plt.show()
    fig_name = tag.split('/')[-1] if '/' in tag else tag
    fig.savefig(f'{fig_name}.pdf', dpi=300)


if __name__ == '__main__':
    summary_fig_plot()      # For reward plot
    # summary_fig_plot(tag="Perf/failed_times", plot_distribution=False)      # For failed times plot
