class IntrospectionLogger:


    def __init__(self):
        self.records = []

    def log_step(self, step_index: int, data: dict):


        self.records.append({
            "step_index": step_index,
            **data
        })
