from __future__ import absolute_import
import torch
import numpy as np
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import _create_text_labels
from torchvision.transforms import functional
from detectron2.data import MetadataCatalog

things_color = [
    (0.5620136544122049, 0.34798600168805305, 0.5101669350565001),
    (0.7050996411400894, 0.4790714684318812, 0.18126723577942236),
    (0.4920739217158292, 0.25734558963187026, 0.5765658668068645),
    (0.7408697051928943, 0.3288411136938306, 0.43075401433146226),
    (0.9149017140183491, 0.16589942427629067, 0.3099211915649369),
    (0.79921628955562, 0.39925182470732773, 0.23770945239727087),
    (0.6580004967376518, 0.36038594257293866, 0.4612648759478487),
    (0.8745465125173779, 0.32970333100685634, 0.16439926153422596),
    (0.6344168668430583, 0.0748376781283183, 0.5954422718351519),
    (0.5946769335344403, 0.07404885587658487, 0.6078363834808271),
    (0.6523388215745849, 0.48683016459111234, 0.26210078418720295),
    (0.646167514379597, 0.3207713243164772, 0.5005885420767506),
    (0.6419006252854835, 0.296095014272689, 0.5195782975493075),
    (0.681736323916992, 0.4368312357472623, 0.3462844260220759),
    (0.7545460308736902, 0.19670088338885727, 0.5043617886658123),
    (0.8366144082057538, 0.22158527604555683, 0.41061292803795696),
    (0.8824937576967857, 0.06998876115350218, 0.40183568089309674),
    (0.772945178088269, 0.413112978624738, 0.26445579360952765),
    (0.5599785537625541, 0.05288092016854584, 0.6166092780878532),
    (0.5468335353980016, 0.531731813978408, 0.19512230073357628),
    (0.8588274733789586, 0.35177319005070895, 0.15637322076226365),
    (0.6611197047996609, 0.2371278042843044, 0.5432888110967016),
    (0.7400940086396335, 0.2792015804598273, 0.47119400407092055),
    (0.7501039229100867, 0.18691549417718073, 0.5114839002811289),
    (0.9242459889810248, 0.16510209536901166, 0.2859449918315642),
    (0.7034500771076568, 0.3171960333236678, 0.47020080297925665),
    (0.804650418474572, 0.09621056632518896, 0.4901135731792768),
    (0.5276916714440919, 0.266508600191752, 0.5686393706182742),
    (0.6288381528373651, 0.48149818608490064, 0.30308553128071886),

    (0.6587385936964898, 0.30789844729510046, 0.5036566750921234),
    (0.8269426777787792, 0.37003574356825103, 0.24098294011389182),
    (0.7528237548739931, 0.18099295387529835, 0.5116377418809095),
    (0.9318628102680226, 0.18652312140316502, 0.23965254217435467),
    (0.826744044906282, 0.07255836614653553, 0.47114707781698356),
    (0.6225552374481869, 0.39242663229111935, 0.447508338956731),
    (0.6156143212204095, 0.47390523886663427, 0.33104685540609635),
    (0.5028669093416189, 0.43562605149309547, 0.4331297367436675),
    (0.8091698305345406, 0.23055700527208134, 0.437844834128994),
    (0.8765949324442198, 0.32800532365681495, 0.1539331270189552),
    (0.8737652014601458, 0.19145827841095947, 0.37503309209882596),
    (0.8032999386580466, 0.3922657126103267, 0.24709348635692435),
    (0.5563044119853887, 0.15624864582919867, 0.6042081588254767),
    (0.5413948594134635, 0.4089840784352307, 0.458013577090074),
    (0.7408153857422185, 0.38123256432917574, 0.371526863208658),
    (0.8383291725994705, 0.13906021287134773, 0.4459051892016492),
    (0.8522536773656169, 0.06635661004078261, 0.442954396780156),
    (0.8380125099971545, 0.22485891395358382, 0.40670097693986007),
    (0.5575134157279324, 0.40538212353172676, 0.457943987358367),
    (0.6514868870633508, 0.26546247155262315, 0.5334276718927493),
    (0.7809280177425519, 0.3777283689402799, 0.32370369114512876),
    (0.9302418050680729, 0.11665216546521222, 0.29780864831438386),
    (0.8786917950112934, 0.3002031622714746, 0.24445411027463848),
    (0.928064362062839, 0.07818589757198834, 0.3149202869565431),
    (0.9612216446044943, 0.1078313021450707, 0.1605623224499679),
    (0.6454970352817397, 0.38550024244461256, 0.44299182445514884),
    (0.7117969077603837, 0.34772655945445835, 0.43715751944352543),
    (0.7461513978909136, 0.3284692755301453, 0.4262790306729816),
    (0.6367749687419464, 0.043769473028327587, 0.5956633193695294),
    (0.9030423530060473, 0.1832924955817049, 0.3253220475982467),
    (0.5200779182087957, 0.3321074941163461, 0.5300188589159174),
    (0.8215431137565952, 0.13041641767450884, 0.46673066121489826),
    (0.6869111090052401, 0.1294907484675056, 0.5659847587781272),
    (0.700844511135411, 0.3862191909920286, 0.4037985184273385),
    (0.7848457438307961, 0.3831827567554937, 0.30750978712335675),
    (0.8093515483708212, 0.3310241033884965, 0.3483987744685417),
    (0.6681789190819015, 0.4860667910832107, 0.24040859366907624),
    (0.8750856714188031, 0.31716095024985097, 0.21551758605037796),
    (0.48959898908265803, 0.5409427314146297, 0.16628009853729572),
    (0.8292027371856674, 0.37752666092174814, 0.20765662909323368),
    (0.81239653264416, 0.3730795842088502, 0.27248992091738705),
    (0.6261694960313245, 0.4833506368470598, 0.3009069164363472),
    (0.556071554786027, 0.3902914437649077, 0.4740454071071492),
    (0.5597508159543112, 0.5184210357122235, 0.24975068396616473),
    (0.5421051806232361, 0.4550659247187539, 0.3986637701938496),
    (0.804244614376991, 0.18471065918974244, 0.4660751347032241),
    (0.8221683569945025, 0.3696786294752878, 0.2558331240254959),
    (0.5409182262854693, 0.38589663542575653, 0.48186459605283916),
    (0.5003425499095077, 0.43421417313702504, 0.43520745247784376),
    (0.7538820369014868, 0.212408161898493, 0.4981223084163867),
    (0.8660354472763635, 0.33049508744293676, 0.2145410419569308)]


class Visualizer_(Visualizer):
    def __init__(self, img, model, threshold=0.5, mode='tensor'):
        """
        Args:
            img: one image (support tensor and rgb np array)
            model: detectron2 model
            threshold: boxes which confidence score low than threshold will not be drawn
        """
        if mode == 'tensor':
            img_rgb = self.tensor2rgb(img)
        else:
            img_rgb = img
        super(Visualizer_, self).__init__(img_rgb, MetadataCatalog.get(model.cfg.DATASETS.TRAIN[0]))
        self.threshold = threshold

    def tensor2rgb(self, img):
        """
        turn a.json tensor float image to a.json uint8 rgb image
        Args:
            img: a.json torch tensor [3,w,h] float type. The image pixel values are between 0 and 1.
        return:
            a.json rgb numpy array
        """
        img = functional.to_pil_image(img)
        img = np.asarray(img)
        return img

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.
        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").
        Returns:
            output (VisImage): image object with visualizations.

        corona: rewrite this function in detectron2. This function will only draw boxes on the image
        """

        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        boxes = boxes[scores >= self.threshold]
        classes = classes[scores >= self.threshold]
        scores = scores[scores >= self.threshold]
        boxes = boxes[classes < 13]
        scores = scores[classes < 13]
        classes = classes[classes < 13]
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        alpha = 0.5
        colors = [
            things_color[c] for c in classes
        ]
        self.overlay_instances(
            boxes=boxes,
            labels=labels,
            alpha=alpha,
            assigned_colors=colors
        )

        return self.output
