import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from collections import Counter
import os
import argparse


def load_models(generator_path, discriminator_path):
    """
    Load the trained generator and discriminator models
    """
    print(f"Loading Generator from: {generator_path}")
    generator = tf.keras.models.load_model(generator_path)

    print(f"Loading Discriminator from: {discriminator_path}")
    discriminator = tf.keras.models.load_model(discriminator_path)

    return generator, discriminator


def generate_samples(generator, num_samples, latent_dim, num_classes):
    """
    Generate samples using the trained generator
    """
    # Generate random noise
    noise = tf.random.normal((num_samples, latent_dim))

    # Generate random class labels
    sampled_labels = tf.random.uniform((num_samples,), minval=0, maxval=num_classes, dtype=tf.int32)

    # Generate fake data
    generated_data = generator.predict([noise, sampled_labels])

    return generated_data, sampled_labels


def evaluate_samples(discriminator, generated_data, sampled_labels, num_classes):
    """
    Evaluate generated samples using the discriminator
    """
    # Convert labels to one-hot encoding
    sampled_labels_onehot = tf.one_hot(sampled_labels, depth=num_classes)

    # Get discriminator predictions
    validity, predicted_classes = discriminator.predict(generated_data)

    # Convert predicted classes to class indices
    predicted_class_indices = np.argmax(predicted_classes, axis=1)

    # Create DataFrame for easier analysis
    results_df = pd.DataFrame({
        'true_label': sampled_labels.numpy(),
        'predicted_label': predicted_class_indices,
        'validity_score': validity.flatten()
    })

    return results_df


def analyze_class_distribution(results_df, num_classes):
    """
    Analyze the class distribution of the generated samples
    """
    # Count the occurrences of each class in true and predicted labels
    true_counts = Counter(results_df['true_label'])
    pred_counts = Counter(results_df['predicted_label'])

    # Create a DataFrame for plotting
    true_df = pd.DataFrame({
        'Class': [f'Class {i}' for i in range(num_classes)],
        'Count': [true_counts.get(i, 0) for i in range(num_classes)],
        'Type': 'Input Label'
    })

    pred_df = pd.DataFrame({
        'Class': [f'Class {i}' for i in range(num_classes)],
        'Count': [pred_counts.get(i, 0) for i in range(num_classes)],
        'Type': 'Predicted Label'
    })

    distribution_df = pd.concat([true_df, pred_df])

    # Calculate accuracy overall and per class
    accuracy = accuracy_score(results_df['true_label'], results_df['predicted_label'])

    # Create confusion matrix
    cm = confusion_matrix(results_df['true_label'], results_df['predicted_label'])

    # Normalize confusion matrix by row (true label)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    return distribution_df, accuracy, cm, cm_norm


def plot_class_distribution(distribution_df, save_path=None):
    """
    Plot the class distribution of generated samples
    """
    plt.figure(figsize=(12, 6))

    # Use seaborn for better visualization
    ax = sns.barplot(x='Class', y='Count', hue='Type', data=distribution_df)

    plt.title('Class Distribution of Generated Network Traffic', fontsize=16)
    plt.xlabel('Class Label', fontsize=14)
    plt.ylabel('Count', fontsize=14)
    plt.xticks(rotation=45)
    plt.legend(title='')

    # Add value labels on bars
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center',
                    xytext=(0, 9),
                    textcoords='offset points')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Class distribution plot saved to {save_path}")

    plt.show()


def plot_confusion_matrix(cm, cm_norm, num_classes, save_path=None):
    """
    Plot the confusion matrix
    """
    # Create class labels
    class_labels = [f'Class {i}' for i in range(num_classes)]

    # Plot raw confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()

    if save_path:
        raw_path = save_path.replace('.png', '_raw.png')
        plt.savefig(raw_path)
        print(f"Raw confusion matrix saved to {raw_path}")

    # Plot normalized confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
    plt.title('Normalized Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.ylabel('True Label', fontsize=14)
    plt.tight_layout()

    if save_path:
        norm_path = save_path.replace('.png', '_norm.png')
        plt.savefig(norm_path)
        print(f"Normalized confusion matrix saved to {norm_path}")

    plt.show()


def plot_validity_distribution(results_df, save_path=None):
    """
    Plot the validity score distribution
    """
    plt.figure(figsize=(10, 6))

    sns.histplot(results_df['validity_score'], bins=20, kde=True)
    plt.title('Distribution of Validity Scores for Generated Samples', fontsize=16)
    plt.xlabel('Validity Score (0=Fake, 1=Real)', fontsize=14)
    plt.ylabel('Count', fontsize=14)

    # Add vertical line for threshold at 0.5
    plt.axvline(x=0.5, color='r', linestyle='--', label='Threshold (0.5)')
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Validity distribution plot saved to {save_path}")

    plt.show()


def analyze_validity_by_class(results_df, num_classes, save_path=None):
    """
    Analyze validity scores by class
    """
    plt.figure(figsize=(12, 6))

    # Create a boxplot of validity scores by true class
    sns.boxplot(x='true_label', y='validity_score', data=results_df)
    plt.title('Validity Scores by Class Label', fontsize=16)
    plt.xlabel('Class Label', fontsize=14)
    plt.ylabel('Validity Score (0=Fake, 1=Real)', fontsize=14)

    # Add horizontal line for threshold at 0.5
    plt.axhline(y=0.5, color='r', linestyle='--', label='Threshold (0.5)')

    # Replace x-axis tick labels
    plt.xticks(range(num_classes), [f'Class {i}' for i in range(num_classes)])

    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Validity by class plot saved to {save_path}")

    plt.show()


def generate_class_specific_samples(generator, num_samples_per_class, num_classes, latent_dim):
    """
    Generate specific number of samples for each class
    """
    all_generated_data = []
    all_labels = []

    for class_idx in range(num_classes):
        # Generate noise
        noise = tf.random.normal((num_samples_per_class, latent_dim))

        # Create array with same class label
        labels = tf.ones((num_samples_per_class,), dtype=tf.int32) * class_idx

        # Generate samples
        generated_data = generator.predict([noise, labels])

        all_generated_data.append(generated_data)
        all_labels.append(labels)

    # Concatenate all generated data and labels
    all_generated_data = np.vstack(all_generated_data)
    all_labels = np.concatenate(all_labels)

    return all_generated_data, all_labels


def analyze_network_traffic_patterns(results_df, generated_data, feature_names=None):
    """
    Analyze patterns in the generated network traffic data
    """
    if feature_names is None:
        # Create default feature names if not provided
        feature_names = [f'Feature_{i}' for i in range(generated_data.shape[1])]

    # Create a DataFrame with the generated data
    gen_df = pd.DataFrame(generated_data, columns=feature_names)

    # Add class information to the DataFrame
    gen_df['true_class'] = results_df['true_label']
    gen_df['predicted_class'] = results_df['predicted_label']
    gen_df['validity_score'] = results_df['validity_score']

    # Calculate statistics for each class
    class_stats = []

    for class_idx in range(len(gen_df['true_class'].unique())):
        class_data = gen_df[gen_df['true_class'] == class_idx].drop(['true_class', 'predicted_class', 'validity_score'],
                                                                    axis=1)

        # Calculate mean and std for each feature
        class_mean = class_data.mean()
        class_std = class_data.std()

        class_stats.append({
            'class': class_idx,
            'means': class_mean,
            'stds': class_std
        })

    return gen_df, class_stats


def main():
    parser = argparse.ArgumentParser(description='Evaluate ACGAN for network traffic generation')
    parser.add_argument('--generator', type=str, required=True, help='Path to the generator model')
    parser.add_argument('--discriminator', type=str, required=True, help='Path to the discriminator model')
    parser.add_argument('--num_samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--latent_dim', type=int, default=256, help='Dimension of the latent space')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--output_dir', type=str, default='./output', help='Directory to save output')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Load models
    generator, discriminator = load_models(args.generator, args.discriminator)

    # Generate random samples
    print(f"Generating {args.num_samples} random samples...")
    generated_data, sampled_labels = generate_samples(generator, args.num_samples, args.latent_dim, args.num_classes)

    # Evaluate samples
    print("Evaluating generated samples...")
    results_df = evaluate_samples(discriminator, generated_data, sampled_labels, args.num_classes)

    # Analyze class distribution
    print("Analyzing class distribution...")
    distribution_df, accuracy, cm, cm_norm = analyze_class_distribution(results_df, args.num_classes)

    # Print overall statistics
    print(f"\nOverall accuracy: {accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(results_df['true_label'], results_df['predicted_label']))

    # Plot results
    print("\nCreating plots...")

    # 1. Plot class distribution
    plot_class_distribution(distribution_df, save_path=os.path.join(args.output_dir, 'class_distribution.png'))

    # 2. Plot confusion matrix
    plot_confusion_matrix(cm, cm_norm, args.num_classes,
                          save_path=os.path.join(args.output_dir, 'confusion_matrix.png'))

    # 3. Plot validity score distribution
    plot_validity_distribution(results_df, save_path=os.path.join(args.output_dir, 'validity_distribution.png'))

    # 4. Plot validity scores by class
    analyze_validity_by_class(results_df, args.num_classes,
                              save_path=os.path.join(args.output_dir, 'validity_by_class.png'))

    # Generate class-specific samples
    print("\nGenerating class-specific samples...")
    num_samples_per_class = 100
    class_samples, class_labels = generate_class_specific_samples(generator, num_samples_per_class, args.num_classes,
                                                                  args.latent_dim)

    # Save generated samples
    samples_df = pd.DataFrame(class_samples)
    samples_df['class'] = class_labels
    samples_df.to_csv(os.path.join(args.output_dir, 'generated_samples.csv'), index=False)
    print(f"Generated samples saved to {os.path.join(args.output_dir, 'generated_samples.csv')}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
