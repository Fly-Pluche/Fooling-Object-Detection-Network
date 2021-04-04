import pyecharts

from tensorboard.backend.event_processing import event_accumulator

old_version1 = ''
old_version2 = ''

FasterX101 = '/Users/keter/Documents/postgraduate_recommendation/Paper/attack/logs/FasterRCNN_20210402-075029_base_500_1000/events.out.tfevents.1617321029.jupyter--44-45-2de3923e06-2d92a7-2d11eb-2db289-2d0255ac100066.11249.0'
ea = event_accumulator.EventAccumulator(path=FasterX101)
ea.Reload()
faster_ap = ea.scalars.Items('ap')
faster_epoch = []
faster_aps = []

for i in faster_ap:
    faster_epoch.append(i.step)
    faster_aps.append(i.value)

RetinaNet = '/Users/keter/Documents/postgraduate_recommendation/Paper/attack/logs/old_version/events.out.tfevents.1616497943.ProfessorRay.18074.0'
ea = event_accumulator.EventAccumulator(path=RetinaNet)
ea.Reload()
RetinaNet_ap = ea.scalars.Items('ap')
RetinaNet_epoch = []
RetinaNet_aps = []

for i in RetinaNet_ap:
    RetinaNet_epoch.append(i.step)
    RetinaNet_aps.append(i.value)

FasterRCNN_Old = '/Users/keter/Documents/postgraduate_recommendation/Paper/attack/logs/old_version2/events.out.tfevents.1616544074.ProfessorRay.5009.0'
ea = event_accumulator.EventAccumulator(path=FasterRCNN_Old)
ea.Reload()
FasterRCNN_Old_ap = ea.scalars.Items('ap')
FasterRCNN_Old_epoch = []
FasterRCNN_Old_aps = []

for i in FasterRCNN_Old_ap:
    FasterRCNN_Old_epoch.append(i.step)
    FasterRCNN_Old_aps.append(i.value)

import pyecharts.options as opts
from pyecharts.charts import Line
from pyecharts.faker import Faker

c = (
    Line()
        .add_xaxis(faster_epoch)
        .add_yaxis(
        "FasterRCNN X101-FPN",
        faster_aps,
        label_opts=opts.LabelOpts(is_show=False),
    )
        .add_xaxis(RetinaNet_epoch)
        .add_yaxis(
        "RetinaNet Old",
        RetinaNet_aps,
        label_opts=opts.LabelOpts(is_show=False),
    )
        .add_xaxis(FasterRCNN_Old_epoch)
        .add_yaxis(
        "FasterRCNN Old",
        FasterRCNN_Old_aps,
        label_opts=opts.LabelOpts(is_show=False),
    )
        .set_global_opts(title_opts=opts.TitleOpts(title="AP"))
        .render("line_markpoint.html")
)
