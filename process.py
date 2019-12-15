import pandas as pd

for i in range(1, 13):
    if i < 10:
        inname = "data/20180" + str(i)
        outname = "data/2018_0" + str(i) + ".csv"
    else:
        inname = "data/2018" + str(i)
        outname = "data/2018_" + str(i) + ".csv"

    df = pd.read_csv(inname)
    default_dict = {'DeleteEvent': 0, 'PullRequestReviewCommentEvent': 0, 'WatchEvent': 0, 'CommitCommentEvent': 0,
    'ReleaseEvent': 0, 'GollumEvent': 0, 'IssueCommentEvent': 0, 'ForkEvent': 0, 'PushEvent': 0, 'CreateEvent': 0,
    'IssuesEvent': 0, 'MemberEvent': 0, 'PullRequestEvent': 0, 'PublicEvent': 0}
    users = {}
    for i, row in df.iterrows():
        if row['actor_id'] not in users:
            users[row['actor_id']] = default_dict.copy()
        users[row['actor_id']][row['type']] = row['n']

    out = pd.DataFrame(users.values())
    print(out.head())
    out.to_csv(outname, index = False)


df = pd.read_csv('data/2018_01.csv')
for i in range(2, 10):
    df_new = pd.read_csv('data/2018_0' + str(i) + '.csv')
    df = pd.concat([df, df_new])
    

for i in range(10, 13):
    df = pd.read_csv('data/2018_' + str(i) + '.csv')
    df = pd.concat([df, df_new])

df.to_csv('data/total.csv', index = False)
