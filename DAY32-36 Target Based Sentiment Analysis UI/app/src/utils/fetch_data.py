import pandas as pd
import os


def fetch_reviews_data(product_name="dell vostro 3400"):
    result_dict = []
    product_dict = {}
    data_path = os.path.join(os.getcwd(), 'datasets', 'test_sentiment_comments_data.csv')
    data_df = pd.read_csv(data_path)
    data_df = data_df[data_df.laptop_name == product_name]
    badge_color_id = {'positive': 'badge bg-success',
                      'negative': 'badge bg-danger',
                      'neutral': 'badge bg-warning'}
    for i in data_df.values:
        badge_color = badge_color_id[i[3]]
        comment_data = {'laptop_name': i[0],
                        'user_name': i[1],
                        'comment': i[2],
                        'sentiment': i[3],
                        'specs': i[4],
                        'badge_color': badge_color}
        result_dict.append(comment_data)
    product_dict['name'] = data_df.values[0][0]
    product_dict['specs'] = data_df.values[0][4]
    return result_dict, product_dict
