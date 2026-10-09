class DNSTriggerNode:
    """
    A prototype trigger node for the Digital Nervous System.
    Maps to 'node-trigger' in the React Flow template.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # These widgets will appear on the node in the UI
                "event_id": ("STRING", {"default": "EVT-001"}),
                "signal_type": (["anomaly", "quality_check", "maintenance"], {"default": "quality_check"}),
                "mes_mock_value": ("FLOAT", {"default": 0.85, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    # The exact output schema required by the handoffEdge to 'node-decision'
    RETURN_TYPES = ("STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("event_id", "signal", "confidence")

    # Execution method to call
    FUNCTION = "activate_trigger"

    # Categorization in the node editor menu
    CATEGORY = "DNS/Triggers"

    # Define the governance contract metadata (used by the DNS validation scheduler)
    DNS_CONTRACT = {
        "agentRole": "production_system",
        "authority": ["emit workflow signal"],
        "forbidden": ["perform agent-only action"],
        "governance": {
            "riskThreshold": 0.4,
            "uncertaintyThreshold": 0.4,
            "requiresApproval": False
        }
    }

    def activate_trigger(self, event_id, signal_type, mes_mock_value):
        """
        Execution logic.
        In a real system, this might use the 'mes_read' tool.
        """
        # 1. Execute tool action (e.g., read from Manufacturing Execution System)
        print(f"[DNS Trigger] Reading MES for event {event_id}...")

        # 2. Package the signal
        signal_data = f"Signal generated from MES: {signal_type}"
        confidence = mes_mock_value  # Derived from the tool or model

        # 3. Governance check (optional pre-flight check at runtime)
        if confidence < (1.0 - self.DNS_CONTRACT["governance"]["uncertaintyThreshold"]):
             print("[DNS Trigger WARNING] High uncertainty detected!")

        # 4. Return the payload.
        # The Comfy-style scheduler will automatically pass these three values
        # as inputs to the next node (node-decision) that is connected to these outputs.
        return (event_id, signal_data, confidence)
