# BRUNO IOCHINS GRISCI
# November 23rd, 2023

import pandas as pd

def main():

    df = pd.read_csv('science_diplomacy - science_diplomacy.csv', header=0, index_col=0)
    # affiliation_country X {description + authkeywords + title}
    df = df[['affiliation_country', 'description', 'authkeywords', 'title']]
    df['text'] = 'Create a complete list with all the geographical regions cited in the following text. The list should be in csv format and be in a single row. Any comments you may have about the task should be outside the list.\n\n' + df['description'] + df['authkeywords'] + df['title']
    df = df.dropna()
    print(df)

    answers = []
    for paper in df['text']:
        print(paper)
        print('\n')
        chat = input('Paste answer here: ')
        answers.append(chat)
        print('\n')
        print(chat)
        print('\n\n')

    df['chat'] = answers

    print(df)

if __name__ == '__main__':
    main()