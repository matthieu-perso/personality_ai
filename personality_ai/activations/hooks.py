import torch

class ModelHooksManager:
    def __init__(self, model):
        """
        Initialize the ModelHooksManager with a PyTorch model.
        
        :param model: A PyTorch model instance.
        """
        self.model = model

    def print_hooks(self, module=None):
        """
        Print all forward hooks of the given module. If no module is specified,
        print hooks for the top-level model.
        
        :param module: A module within the PyTorch model.
        """
        if module is None:
            module = self.model

        if hasattr(module, '_forward_hooks'):
            for hook_id, hook in module._forward_hooks.items():
                print(f'Hook ID: {hook_id}, Hook: {hook}')

    def print_all_hooks(self):
        """
        Print hooks for the top-level model and all its submodules.
        """
        self.print_hooks()  # Print hooks for the top-level model
        for name, submodule in self.model.named_modules():
            print(f"Module: {name}")
            self.print_hooks(submodule)

    def remove_all_forward_hooks(self, module=None):
        """
        Remove all forward hooks from the given module and its submodules.
        If no module is specified, it removes hooks from the top-level model and all its submodules.
        
        :param module: A module within the PyTorch model.
        """
        if module is None:
            module = self.model

        if hasattr(module, '_forward_hooks'):
            module._forward_hooks.clear()

        for child in module.children():
            self.remove_all_forward_hooks(child)

# Usage example:
# from personal_ai.model_hooks_manager import ModelHooksManager

# Assuming 'model' is your PyTorch model instance
# manager = ModelHooksManager(model)

# Print hooks for the top-level module and all submodules
# manager.print_all_hooks()

# Remove all forward hooks from the model and its submodules
# manager.remove_all_forward_hooks()