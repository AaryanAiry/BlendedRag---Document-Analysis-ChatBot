class pdfProcessingError(Exception):
    def __init__(self, message="Failed to process PDF"):
        self.message = message
        super().__init__(self.message)