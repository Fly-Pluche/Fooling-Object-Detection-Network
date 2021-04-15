import pyecharts

from tensorboard.backend.event_processing import event_accumulator

old_version1 = ''
old_version2 = ''

FasterX101 = '/Users/keter/Documents/postgraduate_recommendation/Paper/attack/logs/20210404-211742_base_FasterRCNN/events.out.tfevents.1617542262.jupyter--44-45-2de3923e06-2d92a7-2d11eb-2db289-2d0255ac100066.2020.0'
ea = event_accumulator.EventAccumulator(path=FasterX101)
ea.Reload()
faster_ap = ea.scalars.Items('ap')
faster_epoch = []
faster_aps = []
num = 90
k = 0
for i in faster_ap:
    if k > num:
        break
    k += 1
    faster_epoch.append(i.step)
    faster_aps.append(i.value)

print(faster_epoch)

RetinaNet = '/Users/keter/Documents/postgraduate_recommendation/Paper/attack/logs/old_version/events.out.tfevents.1616497943.ProfessorRay.18074.0'
ea = event_accumulator.EventAccumulator(path=RetinaNet)
ea.Reload()
RetinaNet_ap = ea.scalars.Items('ap')
RetinaNet_epoch = []
RetinaNet_aps = []

k = 0
for i in RetinaNet_ap:
    if k > num:
        break
    k += 1
    RetinaNet_epoch.append(i.step)
    RetinaNet_aps.append(i.value)

FasterRCNN_Old = '/Users/keter/Documents/postgraduate_recommendation/Paper/attack/logs/old_version2/events.out.tfevents.1616544074.ProfessorRay.5009.0'
ea = event_accumulator.EventAccumulator(path=FasterRCNN_Old)
ea.Reload()
FasterRCNN_Old_ap = ea.scalars.Items('ap')
FasterRCNN_Old_epoch = []
FasterRCNN_Old_aps = []

k = 0
for i in FasterRCNN_Old_ap:
    if k > num:
        break
    k += 1
    FasterRCNN_Old_epoch.append(i.step)
    FasterRCNN_Old_aps.append(i.value)

RetinaNet_New = '/Users/keter/Documents/postgraduate_recommendation/Paper/attack/logs/20210330-123611_base/events.out.tfevents.1617078971.ProfessorRay.40475.0'
ea = event_accumulator.EventAccumulator(path=RetinaNet_New)
ea.Reload()
RetinaNet_New_ap = ea.scalars.Items('ap')
RetinaNet_New_epoch = []
RetinaNet_New_aps = []

k = 0
for i in RetinaNet_New_ap:
    if k > num:
        break
    k += 1
    RetinaNet_New_epoch.append(i.step)
    RetinaNet_New_aps.append(i.value)

import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.globals import SymbolType
from pyecharts.faker import Faker

c = (
    Line()
        .add_xaxis(faster_epoch)
        .add_yaxis(
        "FasterRCNN X101-FPN(Our)",
        faster_aps,
        symbol="emptyCircle",
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
        .add_xaxis(RetinaNet_New_epoch)
        .add_yaxis(
        "RetinaNet(Our)",
        RetinaNet_New_aps,
        symbol=SymbolType.TRIANGLE,
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
        .add_xaxis(RetinaNet_epoch)
        .add_yaxis(
        "RetinaNet",
        RetinaNet_aps,
        symbol=SymbolType.ROUND_RECT,
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )
        .add_xaxis(FasterRCNN_Old_epoch)
        .add_yaxis(
        "FasterRCNN X101-FPN",
        FasterRCNN_Old_aps,
        symbol=SymbolType.DIAMOND,
        is_smooth=True,
        label_opts=opts.LabelOpts(is_show=False),
    )

        .set_global_opts(
        xaxis_opts=opts.AxisOpts(
            type_="value", splitline_opts=opts.SplitLineOpts(is_show=True),
            name='epoch'
        ),
        yaxis_opts=opts.AxisOpts(
            type_="value",
            axistick_opts=opts.AxisTickOpts(is_show=True),
            splitline_opts=opts.SplitLineOpts(is_show=True),
            name='AP50'
        ),
    )
        .render("line_markpoint.html")
)
