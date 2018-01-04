import subprocess
from rnn import myList1
from n import e


subprocess.call(['python', 'rnn.py'])

subprocess.call(['python', 'n.py'])

if myList1[0]<e:
    print("class1")
else:
    print("class2")
