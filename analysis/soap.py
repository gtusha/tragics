"""SOAP analysis - minimal implementation."""

import numpy as np

class SOAPCalculator:
    """SOAP descriptor calculations."""
    
    def __init__(self, tragics_instance):
        self.tragics = tragics_instance
        self.soap_vec_frames = None
    
    def calculate_soap(self, *args, **kwargs):
        """See full TRAGICS implementation."""
        raise NotImplementedError("SOAP omitted for brevity. See full package.")
    
    def soap_kernel_vector(self, *args, **kwargs):
        raise NotImplementedError("SOAP omitted for brevity. See full package.")
    
    def soap_kernel_matrix(self, *args, **kwargs):
        raise NotImplementedError("SOAP omitted for brevity. See full package.")
    
    def sequential_similarity_selection(self, *args, **kwargs):
        raise NotImplementedError("SOAP omitted for brevity. See full package.")
