import pickle





def evaluation(det_annos,gt_annos,map_name_to_kitti):

    def kitti_eval(eval_det_annos, eval_gt_annos,map_name_to_kitti):
        from pcdet.datasets.kitti.kitti_object_eval_python import eval as kitti_eval
        from pcdet.datasets.kl.kl_object_eval_python import eval as kl_eval
        from pcdet.datasets.kitti import kitti_utils
        kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
        kitti_utils.transform_annotations_to_kitti_format(
            eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
            info_with_fakelidar=False
        )
        kitti_class_names = [map_name_to_kitti[x] for x in class_names]
        kitti_class_names_unique = list(set(kitti_class_names))
        ap_result_str, ap_dict = kl_eval.get_kl_eval_result(
            gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names_unique
        )
        return ap_result_str, ap_dict


    map_name_to_kitti = {
        # 行人相关
        'Pedestrian': 'Pedestrian',
        'Cone': 'Cone',  # 锥桶尺寸接近行人
        
        # 轿车类
        'Car': 'Car',
        'IGV-Full': 'IGV-Full',     # 智能引导车
        'IGV-Empty': 'IGV-Empty',    # 空载引导车
        'OtherVehicle': 'OtherVehicle', # 其他车辆
        
        # 卡车类（独立类别）
        'Truck': 'Truck',                  # 卡车
        'Trailer-Empty': 'Trailer-Empty',          # 空拖车
        'Trailer-Full': 'Trailer-Full',           # 满载拖车
        'Lorry': 'Lorry',                  # 货车
        'ContainerForklift': 'ContainerForklift',      # 集装箱叉车
        
        # 工程车辆
        'Crane': 'Crane',                    # 起重机
        'Forklift': 'Forklift',                 # 普通叉车
        'ConstructionVehicle': 'ConstructionVehicle'       # 工程车
    }

    ap_result_str, ap_dict=kitti_eval(eval_det_annos,eval_gt_annos,map_name_to_kitti)
    return ap_result_str, ap_dict


if __name__ == '__main__':
    with open('eval_det_annos.pkl', 'rb') as f:
        eval_det_annos = pickle.load(f)
    with open('eval_gt_annos.pkl', 'rb') as f:
        eval_gt_annos = pickle.load(f)
    class_names=['Pedestrian','Car', 'IGV-Full', 'Truck', 'Trailer-Empty',
              'Trailer-Full', 'IGV-Empty','Crane','OtherVehicle', 'Cone',
                    'ContainerForklift', 'Forklift', 'Lorry', 'ConstructionVehicle']
    ap_result_str, ap_dict=evaluation(eval_det_annos,eval_gt_annos,class_names)
    print(ap_result_str)
    # logger.info(result_str)
    # ret_dict.update(result_dict)

    # logger.info('Result is saved to %s' % result_dir)
    # logger.info('****************Evaluation done.*****************')
    # return ret_dict
