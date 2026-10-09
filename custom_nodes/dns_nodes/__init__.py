from .trigger_node import DNSTriggerNode

NODE_CLASS_MAPPINGS = {
    "DNSTriggerNode": DNSTriggerNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DNSTriggerNode": "Production System Trigger"
}

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
