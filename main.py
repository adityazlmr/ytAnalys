from flask import Flask, render_template, request, send_file
from crawling import get_unique_comments, create_csv
from modeling import modeling
from preprocess import preprocess_comments
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os

app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = 'uploads'

# Ensure the "uploads" directory exists
uploads_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(uploads_dir, exist_ok=True)

df = pd.DataFrame()

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/crawling', methods=['GET', 'POST'])
def crawling():
    comments = []  # Initialize comments variable here
    video_title = None
    channel_title = None

    if request.method == 'POST':
        youtube_url = request.form.get('youtube_url')  # Get URL from the form
        if youtube_url:
            video_title,channel_title, comments = get_unique_comments(youtube_url)
            create_csv(comments)
            return render_template('crawling_result.html', comments=comments, video_title=video_title, channel_title=channel_title)
        else:
            error_message = "Please provide a valid YouTube URL."
            return render_template('crawling.html', error=error_message)
    return render_template('crawling.html')


@app.route('/download_csv', methods=['GET'])
def download_csv():
    csv_file = 'comments.csv'  # Ganti dengan nama file CSV yang benar
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
        comments = df.to_dict('records')

        if comments:
            return send_file(csv_file, as_attachment=True)
        else:
            return "No comments available for download."
    else:
        return "File does not exist."




@app.route('/preprocess', methods=['GET', 'POST'])
def preprocess():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            # Save the uploaded file
            file_path = os.path.join(uploads_dir, file.filename)
            file.save(file_path)

            # Process the file in chunks
            chunksize = 1000  # Define your chunk size
            df_chunks = pd.read_csv(file_path, chunksize=chunksize)

            # Initialize an empty list to hold processed chunks
            processed_chunks = []

            # Track number of rows before preprocessing
            rows_before_preprocessing = 0

            # Iterate over chunks and preprocess each chunk
            for chunk in df_chunks:
                # Count number of rows before preprocessing
                rows_before_preprocessing += len(chunk)

                # Perform preprocessing on 'komentar' column
                chunk['komentar_preprocessing'] = chunk['komentar'].apply(preprocess_comments)
                chunk['komentar_preprocessing'].replace('', np.nan, inplace=True)  # Replace empty strings with NaN

                # Drop rows with NaN in 'Komentar_Preprocessing'
                chunk.dropna(subset=['komentar_preprocessing'], inplace=True)

                # Append the processed chunk to the list
                processed_chunks.append(chunk)

            # Concatenate all processed chunks into a single DataFrame
            df = pd.concat(processed_chunks)

            # Save the preprocessed data to a new CSV file
            preprocessed_file_path = os.path.join(uploads_dir, 'preprocessed_comments.csv')
            df.to_csv(preprocessed_file_path, index=False)

            # Calculate number of rows after preprocessing
            rows_after_preprocessing = len(df)

            return render_template('preprocess_result.html', data=df.to_dict('records'),
                                   rows_before_preprocessing=rows_before_preprocessing,
                                   rows_after_preprocessing=rows_after_preprocessing)
        else:
            error_message = "Please upload a valid CSV file."
            return render_template('preprocess.html', error=error_message)
    return render_template('preprocess.html')


@app.route('/download_preprocessed_csv', methods=['GET'])
def download_preprocessed_csv():
    preprocessed_file_path = os.path.join(uploads_dir, 'preprocessed_comments.csv')
    if os.path.exists(preprocessed_file_path):
        return send_file(preprocessed_file_path, as_attachment=True)
    else:
        return "File does not exist."





@app.route('/label', methods=['GET'])
def label():
    preprocessed_file_path = os.path.join(uploads_dir, 'preprocessed_comments.csv')
    if os.path.exists(preprocessed_file_path):
        df = pd.read_csv(preprocessed_file_path)
        return render_template('label.html', data=df.to_dict('records'), preprocessed_file_path=preprocessed_file_path)
    else:
        return "Preprocessed file does not exist."

@app.route('/label_comments', methods=['POST'])
def label_comments():
    file_path = request.form['file_path']
    df = pd.read_csv(file_path)

    for index, row in df.iterrows():
        label_key = 'label_' + str(index + 1)
        label = request.form.get(label_key)
        df.at[index, 'label'] = label

    labeled_file_path = os.path.join(uploads_dir, 'labeled_comments.csv')
    df.to_csv(labeled_file_path, index=False)

    return send_file(labeled_file_path, as_attachment=True)





@app.route('/split', methods=['GET', 'POST'])
def split():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            labeled_df = pd.read_csv(file)

            # Split data menjadi data training dan testing secara random
            train_df, test_df = train_test_split(labeled_df, test_size=0.1, random_state=42)

            # Simpan data training dan testing ke file CSV
            train_file_path = os.path.join(uploads_dir, 'train_data.csv')
            test_file_path = os.path.join(uploads_dir, 'test_data.csv')

            train_df.to_csv(train_file_path, index=False)
            test_df.to_csv(test_file_path, index=False)

            return render_template('split_result.html', train_file_path=train_file_path, test_file_path=test_file_path)

    return render_template('split.html')

@app.route('/download_train_data', methods=['GET'])
def download_train_data():
    train_file_path = request.args.get('train_file_path')
    return send_file(train_file_path, as_attachment=True)

@app.route('/download_test_data', methods=['GET'])
def download_test_data():
    test_file_path = request.args.get('test_file_path')
    return send_file(test_file_path, as_attachment=True)




@app.route('/modeling', methods=['GET', 'POST'])
def modeling_route():
    return modeling()



if __name__ == '__main__':
    app.run(debug=True)
