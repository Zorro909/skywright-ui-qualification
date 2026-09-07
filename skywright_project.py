"""Real AMD workload for the isolated Skywright UI qualification."""

import json
import time

from streaming.base.format.mds.reader import MDSReader

import torch


def train(context):
    assert context.accelerator.kind == 'rocm'
    assert torch.version.hip and torch.cuda.is_available()
    device = torch.cuda.get_device_properties(context.accelerator.index)
    assert '7900 XTX' in device.name, device.name
    model = torch.nn.Linear(64, 64).to(context.accelerator.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
    context.register_checkpoint_state('model', model)
    context.register_checkpoint_state('optimizer', optimizer)
    context.start()
    print(json.dumps({'event': 'gpu-started', 'device': device.name,
                      'memoryBytes': device.total_memory, 'step': context.step,
                      'hip': torch.version.hip}), flush=True)
    total = context.configuration['project']['steps']
    delay = context.configuration['project']['delaySeconds']
    while context.step < total:
        batches = iter(context.dataset.batches(context.dataset_cursor))
        try:
            for batch in batches:
                values = [float(item.ordinal % 64) / 64 for item in batch.items]
                inputs = torch.tensor(values, device=context.accelerator.device)[:, None].expand(-1, 64)
                optimizer.zero_grad()
                loss = model(inputs).square().mean()
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                assert torch.isfinite(loss).item()
                context.commit_step(batch)
                print(json.dumps({'event': 'gpu-step', 'step': context.step,
                                  'loss': loss.item()}), flush=True)
                if context.step >= total:
                    return
                time.sleep(delay)
        finally:
            batches.close()


if __name__ == '__main__':
    import inspect
    inspect.signature(train).bind(object())
    assert torch.version.hip, 'The published project must use the ROCm profile'
    print(f'{MDSReader.__name__}, fixed entry point and ROCm imports passed')
