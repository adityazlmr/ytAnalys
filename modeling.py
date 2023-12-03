from flask import render_template, request
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from collections import Counter
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier



def modeling():
    if request.method == 'POST':
        file_training = request.files['file_training']
        file_testing = request.files['file_testing']

        chunk_size = 1000  # Adjust the chunk size based on your available memory

        if file_training and file_testing and file_training.filename.endswith('.csv') and file_testing.filename.endswith('.csv'):
            # Read training data in chunks
            chunks_training = pd.read_csv(file_training, chunksize=chunk_size)
            df_training = pd.concat(chunks_training, ignore_index=True)

            # Read testing data in chunks
            chunks_testing = pd.read_csv(file_testing, chunksize=chunk_size)
            df_testing = pd.concat(chunks_testing, ignore_index=True)

            # Calculate counts for testing data
            positive_count_test = df_testing[df_testing['label'] == 'positive'].shape[0]
            negative_count_test = df_testing[df_testing['label'] == 'negative'].shape[0]

            # Calculate counts for training data
            positive_count = df_training[df_training['label'] == 'positive'].shape[0]
            negative_count = df_training[df_training['label'] == 'negative'].shape[0]

            total = df_training.shape[0]
            total_test = df_testing.shape[0]

            # Preprocessing
            df_training['komentar_preprocessing'].fillna('', inplace=True)
            df_testing['komentar_preprocessing'].fillna('', inplace=True)

            # CountVectorizer on training data
            vectorizer = CountVectorizer()
            X_training = vectorizer.fit_transform(df_training['komentar_preprocessing'])

            # Get the vocabulary from training data
            vocabulary_training = vectorizer.get_feature_names_out()

            # Transform testing data using the same vocabulary
            X_testing = vectorizer.transform(df_testing['komentar_preprocessing'])

            # Euclidean distances
            distances = euclidean_distances(X_testing, X_training)

            # KNN
            k = 7  # Nilai K untuk K-Nearest Neighbors
            closest_labels = []
            matching_indices = []  # Define matching_indices here

            for i in range(len(distances)):
                indices = np.argsort(distances[i])[:k]
                matching_indices.append(indices.tolist())
                labels = df_training['label'].iloc[indices].values
                most_common_label = Counter(labels).most_common(1)[0][0]
                closest_labels.append(most_common_label)

            # Hasil prediksi dari KNN
            predicted_labels = pd.Series(closest_labels)

            # Confusion Matrix
            cm = confusion_matrix(df_testing['label'], predicted_labels, labels=['positive', 'negative'])

            accuracy_percentage = accuracy_score(df_testing['label'], predicted_labels) * 100
            precision = precision_score(df_testing['label'], predicted_labels, pos_label='positive') * 100
            recall = recall_score(df_testing['label'], predicted_labels, pos_label='positive') * 100
            f1_score_value = f1_score(df_testing['label'], predicted_labels, pos_label='positive') * 100

            # Menyimpan nama file grafik yang dihasilkan
            chart_image = create_label_chart(positive_count, negative_count,
                                             positive_count_test, negative_count_test)

            piechart_image = create_pie_chart(df_testing['label'], predicted_labels)

            # Render template with the matching indices
            return render_template('modeling_result.html',
                                    positive_count=positive_count,
                                    negative_count=negative_count,
                                    total=total,
                                    total_test=total_test,
                                    word_frequency_matrix_training=pd.DataFrame(X_training.toarray(), columns=vectorizer.get_feature_names_out()),
                                    word_frequency_matrix_testing=pd.DataFrame(X_testing.toarray(), columns=vocabulary_training),
                                    euclidean_distance=distances,
                                    data=df_training.to_dict('records'),
                                    data_training=df_training.to_dict('records'),
                                    data_testing=df_testing.to_dict('records'),
                                    min_length_training=min(len(df_training), X_training.shape[0]),
                                    min_length_testing=len(df_testing),
                                    matching_indices=matching_indices,  # Include matching_indices here
                                    confusion_matrix=cm,
                                    accuracy_percentage=accuracy_percentage,
                                    precision=precision,
                                    recall=recall,
                                    f1_score=f1_score_value,
                                    chart_image=chart_image,
                                    comparison_image=piechart_image,
                                    predicted_labels=predicted_labels)

    return render_template('modeling.html')

    
def create_label_chart(positive_count_train, negative_count_train,
                       positive_count_test, negative_count_test):
    labels = ['Positive', 'Negative']
    total_training_counts = [positive_count_train, negative_count_train]
    total_testing_counts = [positive_count_test, negative_count_test]

    # Calculate total counts combining training and testing
    total_counts = [train + test for train, test in zip(total_training_counts, total_testing_counts)]

    x = np.arange(len(labels))  # Adjusting bar positions
    width = 0.25  # Reducing the width of bars

    fig, ax = plt.subplots()
    rects3 = ax.bar(x - width, total_counts, width, label='Total', alpha=0.5, hatch='///', color='gray', edgecolor='black')  # Add total counts bars
    rects1 = ax.bar(x, total_training_counts, width, label='Training', color='blue', edgecolor='black')
    rects2 = ax.bar(x + width, total_testing_counts, width, label='Testing', color='orange', edgecolor='black')

    ax.set_xlabel('Labels')
    ax.set_ylabel('Counts')
    ax.set_title('Label Counts in Training and Testing Data')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add counts inside the bars for training and testing data
    for rects in [rects1, rects2]:
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # Offset for the text inside the bar
                        textcoords="offset points",
                        ha='center', va='bottom', color='black')

    # Add counts on top of the total bars
    for rect in rects3:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # Offset for the text on top of the bar
                    textcoords="offset points",
                    ha='center', va='bottom', color='black')

    # Mengambil nilai maksimum dari total_counts sebagai nilai maksimum sumbu Y
    max_value = max(total_counts)

    # Menambahkan margin ke nilai maksimum sumbu Y
    ax.set_ylim(0, max_value * 1.2)  # Menaikkan batas atas sebesar 20% dari nilai maksimum

    # Save the plot as an image
    plt.savefig('static/label_chart.png', bbox_inches='tight')  # Simpan grafik sebagai file gambar

    # Return the filename to be used in HTML
    return 'label_chart.png'




def create_pie_chart(actual_labels, predicted_labels):
    # Count the occurrences of each label
    predicted_counts = predicted_labels.value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))  # Ukuran sesuai dengan label chart

    # Definisi warna untuk setiap label
    colors = ['#4E9F3D',  '#950101']

    # Create pie chart for predicted labels dengan warna yang telah didefinisikan
    wedges, _, autotexts = ax.pie(predicted_counts, labels=predicted_counts.index, autopct='%1.1f%%', textprops=dict(color="w"), colors=colors)

    # Menambahkan jumlah label pada pie chart
    label_str = [f'{index}: {value}' for index, value in predicted_counts.items()]
    ax.legend(wedges, label_str, title='Label Counts', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1))

    ax.set_title('Labels Distribution')

    # Save the plot as an image
    plt.savefig('static/pie_chart.png', bbox_inches='tight')

    # Return the filename to be used in HTML
    return 'pie_chart.png'

