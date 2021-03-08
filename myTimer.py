class myTimer:
    """
    This is a simple class to set up a timer object that times the duration of a cell.

    Inputs:
        - time.time() object.
    
    Outputs:
        - Prints time cell took to run.
    """
    def __init__(self, startTime):
        from datetime import datetime
        self.startTime = startTime
        print("Cell started at: {}".format(datetime.now().ctime()))
        
    def getCellTime(self, endTime):
        import math
        duration = endTime - self.startTime
        hrs = math.floor(duration/3600)
        mins = math.floor((duration - 3600*hrs)/60)
        secs = duration - 3600 * hrs - 60 * mins
        print("\nThis cell took {} hrs, {} mins and {:.2f} seconds to run.".format(hrs, mins, secs))