import pandas as pd

import matplotlib.pyplot as plt


act = ['WatchEvent', 'IssueCommentEvent','IssuesEvent', 'MemberEvent','PullRequestEvent','ForkEvent']
cont = ['CreateEvent', 'DeleteEvent', 'PushEvent', 'CommitCommentEvent', 'ReleaseEvent','PublicEvent', 'PullRequestReviewCommentEvent', 'GollumEvent']
sm = []
sact = []
scont = []

for i in range(1, 10):
    df = pd.read_csv('data/2018_0' + str(i) + '.csv')
    sm.append(sum([sum(df[col]) for col in df.columns.values]))
    sact.append(sum([sum(df[col]) for col in act]))
    scont.append(sum([sum(df[col]) for col in cont]) - 2 * sum(df['DeleteEvent']))

for i in range(10, 13):
    df = pd.read_csv('data/2018_' + str(i) + '.csv')
    sm.append(sum([sum(df[col]) for col in df.columns.values]))
    sact.append(sum([sum(df[col]) for col in act]))
    scont.append(sum([sum(df[col]) for col in cont]) - 2 * sum(df['DeleteEvent']))

print(sm)
print(sact)
print(scont)

month = list(range(1, 13))

plt.figure(figsize=(12,8))
l1=plt.plot(month,sm,'r--',label='number of events')
l2=plt.plot(month,sact,'g--',label='activities')
l3=plt.plot(month,scont,'b--',label='contributions')

plt.plot(month,sm,'ro-',month,sact,'g+-',month,scont,'b^-')
plt.title('Events in different months')
plt.xlabel('month')
plt.ylabel('value')

plt.legend()
plt.show()
