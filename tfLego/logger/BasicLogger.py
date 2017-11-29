class BasicLogger:

    def __init__(self):

        self.loss_history = []
        self.accuracy_history = []

        self.val_loss_history = []
        self.val_accuracy_history = []

        self.initialise()

    def initialise(self):

        self.total_loss = 0
        self.total_accuracy = 0
        self.current = 0


    def log_batch(self, loss, outputs, accuracy, *args, **kwargs):

        self.current += 1
        self.total_loss += loss
        self.total_accuracy += accuracy

    def log_epoch(self, i, X, is_val=False, *args, **kwargs):

        loss = self.total_loss / len(X)
        accuracy = self.total_accuracy / len(X)

        if(is_val):
            self.val_loss_history.append(loss)
            self.val_accuracy_history.append(accuracy)
        else:
            self.loss_history.append(loss)
            self.accuracy_history.append(accuracy)

        print('EPOCH: {0}. AVG Loss: {1:0.4f} Acc: {2:0.4f}'.format(i,loss, accuracy))

        self.initialise()


