class SubmissionOutput:

    def __init__(self,stdout):

        self.__stdout = stdout
    
    @property
    def stdout(self):
        return self.__stdout

    @property
    def job_number(self):
        return self.stdout.split('.')[0]

    @property
    def jobID(self):
        return self.stdout
    

def main():

    output = SubmissionOutput('2498312.sched-torque.pace.gatech.edu')
    print(output.jobID)

if __name__ == '__main__':
    main()

