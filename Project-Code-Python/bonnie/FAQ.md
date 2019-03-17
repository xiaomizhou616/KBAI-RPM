#Bonnie FAQ?


## What is Bonnie?
Bonnie is the front-end webserver to the automatic grader that Udacity designed and built for the GT OMSCS program.

## How do I submit my code?
One way or another a submission method will be provided to you along with the assignment.  Most often this takes the form of a command-line script `submit.py`.  Consult the assignment instructions for details.

## What software dependencies are ther for submitting my code?

* python2 or python3
* requests
* future

## Can I login to Bonnie directly to confirm that my submission was received?
Yes. Login to [https://bonnie.udacity.com](https://bonnie.udacity.com) and visit the Student Portal.

## Can I submit more than once?
Usually, yes. A professor or TA can rate-limit your submissions, but most are generous in this respect.

## How do I find out if one of my assignments is rate-limited?
This should is available from the Student Portal on the [bonnie webserver](https://bonnie.udacity.com).

## What is a jwt?
jwt stands for [JSON Web Token](https://jwt.io/).  Some submission method may allow you to save a jwt to your local filesystem so that you don't have to provide your username and password each time you make a submission.

If you have two-factor identification enabled for GT, you may need to download manually a jwt from the [bonnie webserver](https://bonnie.udacity.com) CHANGE TO PROPER URL to be able to submit your code.

## If I submit more than once, which submission will be graded?
Your last submission before the deadline will graded.

## How do I know what the deadline is for an assignment?
You should consult the syllabus or schedule for your course.  The grading service doesn't store deadlines per se, but rather allows the TA to specify an upper bound on the date when he pulls submissions and results from the webserver.

## Can I submit after the deadline specified by the course?
Yes, but it will only be considered at the Prof/TA's discretion.

## What do I do if the message returned after I submit doesn't make any sense to me?
Usually, you will want to ask the TAs in the class forum.

If you suspect that an error is occurring in the infrastructure itself, then login to the [webserver](https://bonnie.udacity.com) and find the result associated with your submission.  If the "error_report" field is not null, then there was a server error.  Please forward the full object you see in the student portal to one of your TAs.



