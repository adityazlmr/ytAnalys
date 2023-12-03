from googleapiclient.discovery import build

def get_unique_comments(video_url):
    api_key = 'AIzaSyDGvPq3nwcTd2EANHURxMS5DjoOp27_MOk'
    youtube = build('youtube', 'v3', developerKey=api_key)

    video_id = video_url.split('=')[-1]
    video_info = youtube.videos().list(
        part="snippet",
        id=video_id
    ).execute()

    video_title = video_info['items'][0]['snippet']['title'] if video_info['items'] else 'Unknown'
    channel_title = video_info['items'][0]['snippet']['channelTitle'] if video_info['items'] else 'Unknown'
    comments = []

    nextPageToken = None
    while True:
        response = youtube.commentThreads().list(
            part='snippet',
            videoId=video_id,
            maxResults=100,
            pageToken=nextPageToken
        ).execute()

        for item in response['items']:
            comment = item['snippet']['topLevelComment']['snippet']
            comments.append({
                'author': comment['authorDisplayName'],
                'date': comment['publishedAt'],
                'text': comment['textDisplay']
            })

        nextPageToken = response.get('nextPageToken')
        if not nextPageToken:
            break  # Break the loop if there are no more pages

    # Code untuk menghapus komentar duplikat
    unique_comments = [dict(t) for t in {tuple(d.items()) for d in comments}]
    return video_title,channel_title, unique_comments


def create_csv(comments):
    import csv

    with open('comments.csv', 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['nomor', 'nama_akun', 'tanggal_komentar', 'komentar']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for i, comment in enumerate(comments, 1):
            writer.writerow({
                'nomor': i,
                'nama_akun': comment['author'],
                'tanggal_komentar': comment['date'],
                'komentar': comment['text']
            })

