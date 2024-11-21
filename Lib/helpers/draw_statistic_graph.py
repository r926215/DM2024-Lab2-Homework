import matplotlib.pyplot as plt
import seaborn as sns
import os
import _const as const

def Compare_Features_Occurrences(before_data, current_data=None):
    # Calculate feature occurrences for X_train_data and X_test_data
    before_removing_feature_count = before_data.count()


    if current_data is not None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(16, 6), sharey=True)
        # Plot occurrences for before_removing_data
        before_removing_feature_count.plot(kind='bar', color='skyblue', ax=axes[0])
        axes[0].set_title(f'Occurrences of Each Feature Before removing_outliers (Data Length: {len(before_data)})')
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Occurrences')
        axes[0].tick_params(axis='x', rotation=90)

        # Plot occurrences for current_data
        after_removing_train_feature_count = current_data.count()
        after_removing_train_feature_count.plot(kind='bar', color='lightcoral', ax=axes[1])
        axes[1].set_title(f'Occurrences of Each Feature After removing_outliers (Data Length: {len(current_data)})')
        axes[1].set_xlabel('Features')
        axes[1].tick_params(axis='x', rotation=90)
    else:
        # Create a figure with only one subplot if `cur_data` is not provided
        fig, ax = plt.subplots(figsize=(8, 6))
        before_removing_feature_count.plot(kind='bar', color='skyblue', ax=ax)
        ax.set_title(f'Occurrences of Each Feature (Data Length: {len(before_data)})')
        ax.set_xlabel('Features')
        ax.set_ylabel('Occurrences')
        ax.tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def Feature_Values_Distribution(before_data, current_data=None):

    if current_data is not None:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
        sns.boxplot(data=before_data, ax=axes[0])
        axes[0].set_title("Before: Feature Values Distribution")
        axes[0].tick_params(axis='x', rotation=90)

        sns.boxplot(data=current_data, ax=axes[1])
        axes[1].set_title("After: Feature Values Distribution")
        axes[1].tick_params(axis='x', rotation=90)
    else:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        sns.boxplot(data=before_data, ax=ax)
        ax.set_title("Feature Values Distribution")
        ax.tick_params(axis='x', rotation=90)


    plt.tight_layout()
    plt.show()

def Features_Occurrences(current_data):

    fig, ax = plt.subplots(figsize=(8, 6))
    current_data.plot(kind='bar', color='skyblue', ax=ax)
    ax.set_title(f'Occurrences of Each Feature (Total Occurrence: {current_data.sum()})')
    ax.set_xlabel('Features')
    ax.set_ylabel('Occurrences')
    ax.tick_params(axis='x', rotation=90)

    # Adjust layout
    plt.tight_layout()
    plt.show()

def Descriptive_Statistics(current_data):

    current_data.describe().to_csv(os.path.join(const.dir_path['descriptive_statistics'], 'statistics.csv'), index=False)

    correlation_matrix = current_data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

    print(current_data.describe())

def Explained_Variance_Ratio(variance_ratio):

    plt.figure(figsize=(8, 5))
    plt.bar(range(1, len(variance_ratio) + 1), variance_ratio, color='skyblue')
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Component')
    plt.xticks(range(1, len(variance_ratio) + 1))

    plt.show()

def Dimensionality_Reduction_Visualization(data, target):
    plt.figure(figsize=(8, 6))
    plt.scatter(data[:, 0], data[:, 1], c=target, cmap='viridis', s=50, edgecolors='k')
    plt.colorbar(label='Classes')
    plt.title('t-SNE visualization of dataset')
    plt.xlabel('t-SNE component 1')
    plt.ylabel('t-SNE component 2')
    plt.show()