# ComfyUI Widgets: A Guide for Codex

When parsing or generating ComfyUI workflow JSONs, it is critical to understand the distinction between **Inputs**, **Outputs**, and **Widgets**.

## 1. What are Widgets?

In ComfyUI (built on top of LiteGraph.js), a **Widget** is a UI control embedded directly on the face of a node. Examples include:
- A text box for a prompt (e.g., `CLIPTextEncode`).
- A slider for a float/int (e.g., `EmptyLatentImage` dimensions).
- A dropdown menu to select a model (e.g., `UNETLoader`).
- A toggle switch (boolean).

While **Inputs** (`inputs` array in JSON) receive data from the output link of *another* upstream node, **Widgets** hold static or user-defined values stored directly inside the node itself.

## 2. How Widgets are Serialized in JSON

When you export a workflow to JSON, you will see a `widgets_values` array inside the node object.

```json
{
  "id": 2,
  "type": "CLIPTextEncode",
  "inputs": [{"name": "clip", "type": "CLIP", "link": 1}],
  "outputs": [{"name": "CONDITIONING", "type": "CONDITIONING", "links": [2], "slot_index": 0}],
  "widgets_values": ["A beautiful sunset over mountains, photorealistic"]
}
```

Notice that `widgets_values` is an **Array**, not a dictionary. The values are stored strictly in the order they were defined in the Python backend.

## 3. How Widgets map to the Python Backend

In the ComfyUI backend (`nodes.py` or your custom nodes), the `INPUT_TYPES` class method defines both inputs and widgets.

```python
class EmptyLatentImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                # These three are WIDGETS because they define primitive types
                "width": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 8}),
                "height": ("INT", {"default": 512, "min": 16, "max": 8192, "step": 8}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 4096})
            }
        }
```
If a parameter in `INPUT_TYPES` uses a primitive definition like `"INT"`, `"FLOAT"`, `"STRING"`, `"BOOLEAN"`, or a `list` of strings (dropdown), it becomes a **Widget**.
If a parameter uses a custom type (like `"MODEL"`, `"CLIP"`, `"IMAGE"`), it becomes an **Input** link.

Therefore, for the `EmptyLatentImage` node above, the JSON serialization will always look like this:
`"widgets_values": [512, 512, 1]`
* Index 0 maps to `width`
* Index 1 maps to `height`
* Index 2 maps to `batch_size`

## 4. Converting Widgets to Inputs (Advanced)

In the ComfyUI interface, a user can right-click a Widget and select "Convert to Input". This removes the UI control from the node and creates an Input pin instead, allowing the value to be driven by another node (like an Integer primitive node).

If a widget is converted to an input, its value is **removed** from the `widgets_values` array, and it will appear in the `inputs` array instead.

## 5. Summary for Codex Workflow Generation

When generating `.json` files for ComfyUI:
1. Fetch the node's schema (e.g., from the `/object_info` API).
2. Count the primitive types (`INT`, `FLOAT`, `STRING`, `list`) in the `"required"` dict.
3. Construct the `widgets_values` array in the *exact sequential order* those primitives are listed in the schema.
4. Ensure the types match (don't pass a string `"512"` if it expects an integer `512`).
