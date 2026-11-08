"""M4: fixed object, dual-mask local background optimization and scene export."""
import os
from ..core.context import Context,arguments
from ..core.workflow import validate_workflow


def run(ctx):
    validate_workflow(ctx.cfg)
    from ..core.scene_integration import integrate
    integrate(ctx)
    from ..core.local_background_optimizer import optimize_background
    from ..core.m4_outputs import export_m4
    # M4 owns per-view dual-mask/occlusion preparation, then background updates.
    optimize_background(ctx)
    export_m4(ctx)


if __name__=='__main__':
    args=arguments(); ctx=Context(args.config); os.environ.update(ctx.env())
    run(ctx)
