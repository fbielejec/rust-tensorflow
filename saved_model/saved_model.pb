??
??
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?"serve*2.3.02v2.3.0-rc2-23-gb36436b0878??

NoOpNoOp
i
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*%
valueB B


signatures
 
?
serving_default_scoresPlaceholder*0
_output_shapes
:??????????????????*
dtype0*%
shape:??????????????????
[
serving_default_user_indexPlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_scoresserving_default_user_index*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference_signature_wrapper_239
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameConst*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *%
f R
__inference__traced_save_263
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *(
f#R!
__inference__traced_restore_273ݩ
?
i
__inference__traced_save_263
file_prefix
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const?
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_486bbabfeba44d9b9929c3a05343e83d/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: : 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: 
?l
Q
__inference___call___231

scores

user_index
identity??PrintV2?
StringFormatStringFormat*
T
 *
_output_shapes
: *
placeholder{}*.
template" Executing with tensorflow v2.3.02
StringFormatJ
PrintV2PrintV2StringFormat:output:0*
_output_shapes
 2	
PrintV2`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2scores
user_indexGatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:?????????2

GatherV2q
Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
Reshape/shapev
ReshapeReshapeGatherV2:output:0Reshape/shape:output:0*
T0*#
_output_shapes
:?????????2	
ReshapeL
	map/ShapeShapescores*
T0*
_output_shapes
:2
	map/Shape|
map/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
map/strided_slice/stack?
map/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_1?
map/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
map/strided_slice/stack_2?
map/strided_sliceStridedSlicemap/Shape:output:0 map/strided_slice/stack:output:0"map/strided_slice/stack_1:output:0"map/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
map/strided_slice?
map/TensorArrayV2/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2!
map/TensorArrayV2/element_shape?
map/TensorArrayV2TensorListReserve(map/TensorArrayV2/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2?
9map/TensorArrayUnstack/TensorListFromTensor/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2;
9map/TensorArrayUnstack/TensorListFromTensor/element_shape?
+map/TensorArrayUnstack/TensorListFromTensorTensorListFromTensorscoresBmap/TensorArrayUnstack/TensorListFromTensor/element_shape:output:0*
_output_shapes
: *
element_dtype0*

shape_type02-
+map/TensorArrayUnstack/TensorListFromTensorX
	map/ConstConst*
_output_shapes
: *
dtype0*
value	B : 2
	map/Const?
!map/TensorArrayV2_1/element_shapeConst*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!map/TensorArrayV2_1/element_shape?
map/TensorArrayV2_1TensorListReserve*map/TensorArrayV2_1/element_shape:output:0map/strided_slice:output:0*
_output_shapes
: *
element_dtype0*

shape_type02
map/TensorArrayV2_1r
map/while/loop_counterConst*
_output_shapes
: *
dtype0*
value	B : 2
map/while/loop_counter?
	map/whileStatelessWhilemap/while/loop_counter:output:0map/strided_slice:output:0map/Const:output:0map/TensorArrayV2_1:handle:0map/strided_slice:output:0;map/TensorArrayUnstack/TensorListFromTensor:output_handle:0Reshape:output:0*
T
	2*
_lower_using_switch_merge(*
_num_original_outputs*/
_output_shapes
: : : : : : :?????????* 
_read_only_resource_inputs
 *
bodyR
map_while_body_24*
condR
map_while_cond_23*.
output_shapes
: : : : : : :?????????2
	map/while
map/TensorArrayV2Stack/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
map/TensorArrayV2Stack/Const?
&map/TensorArrayV2Stack/TensorListStackTensorListStackmap/while:output:3%map/TensorArrayV2Stack/Const:output:0*#
_output_shapes
:?????????*
element_dtype02(
&map/TensorArrayV2Stack/TensorListStack[
	Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
	Greater/y?
GreaterGreater/map/TensorArrayV2Stack/TensorListStack:tensor:0Greater/y:output:0*
T0*#
_output_shapes
:?????????2	
GreaterM
WhereWhereGreater:z:0*'
_output_shapes
:?????????2
Where{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceWhere:index:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_sliceP
CastCast
user_index*

DstT0	*

SrcT0*
_output_shapes
: 2
Cast?
NotEqualNotEqualstrided_slice:output:0Cast:y:0*
T0	*#
_output_shapes
:?????????*
incompatible_shape_error( 2

NotEqualR
Where_1WhereNotEqual:z:0*'
_output_shapes
:?????????2	
Where_1
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceWhere_1:index:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_1d
GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_1/axis?

GatherV2_1GatherV2strided_slice:output:0strided_slice_1:output:0GatherV2_1/axis:output:0*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????2

GatherV2_1W
IsNanIsNanReshape:output:0*
T0*#
_output_shapes
:?????????2
IsNanO
Where_2Where	IsNan:y:0*'
_output_shapes
:?????????2	
Where_2
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_2/stack?
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_2/stack_1?
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_2/stack_2?
strided_slice_2StridedSliceWhere_2:index:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_2d
GatherV2_2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_2/axis?

GatherV2_2GatherV2/map/TensorArrayV2Stack/TensorListStack:tensor:0GatherV2_1:output:0GatherV2_2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_2d
GatherV2_3/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_3/axis?

GatherV2_3GatherV2scoresGatherV2_1:output:0GatherV2_3/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:??????????????????2

GatherV2_3d
GatherV2_4/axisConst*
_output_shapes
: *
dtype0*
value	B :2
GatherV2_4/axis?

GatherV2_4GatherV2GatherV2_3:output:0strided_slice_2:output:0GatherV2_4/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*0
_output_shapes
:??????????????????2

GatherV2_4k
IsNan_1IsNanGatherV2_4:output:0*
T0*0
_output_shapes
:??????????????????2	
IsNan_1u

zeros_like	ZerosLikeGatherV2_4:output:0*
T0*0
_output_shapes
:??????????????????2

zeros_like?
SelectV2SelectV2IsNan_1:y:0zeros_like:y:0GatherV2_4:output:0*
T0*0
_output_shapes
:??????????????????2

SelectV2Q
ShapeShapeGatherV2_2:output:0*
T0*
_output_shapes
:2
Shapex
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSliceShape:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3h
Reshape_1/shape/0Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape_1/shape/0?
Reshape_1/shapePackReshape_1/shape/0:output:0strided_slice_3:output:0*
N*
T0*
_output_shapes
:2
Reshape_1/shape?
	Reshape_1ReshapeGatherV2_2:output:0Reshape_1/shape:output:0*
T0*'
_output_shapes
:?????????2
	Reshape_1s
MatMulMatMulReshape_1:output:0SelectV2:output:0*
T0*'
_output_shapes
:?????????2
MatMul?
strided_slice_4/stackConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
strided_slice_4/stack|
strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_4/stack_1|
strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_4/stack_2?
strided_slice_4StridedSliceMatMul:product:0strided_slice_4/stack:output:0 strided_slice_4/stack_1:output:0 strided_slice_4/stack_2:output:0*
Index0*
T0*#
_output_shapes
:?????????*
shrink_axis_mask2
strided_slice_4_
Greater_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Greater_1/y
	Greater_1Greaterstrided_slice_4:output:0Greater_1/y:output:0*
T0*#
_output_shapes
:?????????2
	Greater_1S
Where_3WhereGreater_1:z:0*'
_output_shapes
:?????????2	
Where_3
strided_slice_5/stackConst*
_output_shapes
:*
dtype0*
valueB"    ????2
strided_slice_5/stack?
strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_5/stack_1?
strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_5/stack_2?
strided_slice_5StridedSliceWhere_3:index:0strided_slice_5/stack:output:0 strided_slice_5/stack_1:output:0 strided_slice_5/stack_2:output:0*
Index0*
T0	*#
_output_shapes
:?????????*

begin_mask*
end_mask*
shrink_axis_mask2
strided_slice_5d
GatherV2_5/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_5/axis?

GatherV2_5GatherV2strided_slice_4:output:0strided_slice_5:output:0GatherV2_5/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2

GatherV2_5_
ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
ConstV
SumSumReshape_1:output:0Const:output:0*
T0*
_output_shapes
: 2
Sumn
truedivRealDivGatherV2_5:output:0Sum:output:0*
T0*#
_output_shapes
:?????????2	
truediv^
argsort/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
argsort/axisY
argsort/ShapeShapetruediv:z:0*
T0*
_output_shapes
:2
argsort/Shape?
argsort/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
argsort/strided_slice/stack?
argsort/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
argsort/strided_slice/stack_1?
argsort/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
argsort/strided_slice/stack_2?
argsort/strided_sliceStridedSliceargsort/Shape:output:0$argsort/strided_slice/stack:output:0&argsort/strided_slice/stack_1:output:0&argsort/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
argsort/strided_slice^
argsort/RankConst*
_output_shapes
: *
dtype0*
value	B :2
argsort/Rank?
argsort/TopKV2TopKV2truediv:z:0argsort/strided_slice:output:0*
T0*2
_output_shapes 
:?????????:?????????2
argsort/TopKV2o
Cast_1Caststrided_slice_2:output:0*

DstT0*

SrcT0	*#
_output_shapes
:?????????2
Cast_1d
GatherV2_6/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_6/axis?

GatherV2_6GatherV2
Cast_1:y:0argsort/TopKV2:indices:0GatherV2_6/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:?????????2

GatherV2_6d
GatherV2_7/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2_7/axis?

GatherV2_7GatherV2truediv:z:0argsort/TopKV2:indices:0GatherV2_7/axis:output:0*
Taxis0*
Tindices0*
Tparams0*#
_output_shapes
:?????????2

GatherV2_7?
stackPackGatherV2_6:output:0GatherV2_7:output:0*
N*
T0*'
_output_shapes
:?????????*

axis2
stackl
IdentityIdentitystack:output:0^PrintV2*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:??????????????????: 2
PrintV2PrintV2:X T
0
_output_shapes
:??????????????????
 
_user_specified_namescores:B>

_output_shapes
: 
$
_user_specified_name
user_index
?
?
map_while_cond_23$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1$
 map_while_less_map_strided_slice8
4map_while_map_while_cond_23___redundant_placeholder08
4map_while_map_while_cond_23___redundant_placeholder1
map_while_identity
?
map/while/LessLessmap_while_placeholder map_while_less_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less?
map/while/Less_1Less map_while_map_while_loop_countermap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Less_1|
map/while/LogicalAnd
LogicalAndmap/while/Less_1:z:0map/while/Less:z:0*
_output_shapes
: 2
map/while/LogicalAndo
map/while/IdentityIdentitymap/while/LogicalAnd:z:0*
T0
*
_output_shapes
: 2
map/while/Identity"1
map_while_identitymap/while/Identity:output:0*%
_input_shapes
: : : : : ::: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
::

_output_shapes
:
?w
9
__inference_cor_na_omit_136
x
y
identityH
IsNanIsNanx*
T0*#
_output_shapes
:?????????2
IsNanV

LogicalNot
LogicalNot	IsNan:y:0*#
_output_shapes
:?????????2

LogicalNotL
IsNan_1IsNany*
T0*#
_output_shapes
:?????????2	
IsNan_1\
LogicalNot_1
LogicalNotIsNan_1:y:0*#
_output_shapes
:?????????2
LogicalNot_1s
stackPackLogicalNot:y:0LogicalNot_1:y:0*
N*
T0
*'
_output_shapes
:?????????2
stackp
All/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2
All/reduction_indicesf
AllAllstack:output:0All/reduction_indices:output:0*#
_output_shapes
:?????????2
AllY
boolean_mask/ShapeShapex*
T0*
_output_shapes
:2
boolean_mask/Shape?
 boolean_mask/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2"
 boolean_mask/strided_slice/stack?
"boolean_mask/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_1?
"boolean_mask/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice/stack_2?
boolean_mask/strided_sliceStridedSliceboolean_mask/Shape:output:0)boolean_mask/strided_slice/stack:output:0+boolean_mask/strided_slice/stack_1:output:0+boolean_mask/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask/strided_slice?
#boolean_mask/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2%
#boolean_mask/Prod/reduction_indices?
boolean_mask/ProdProd#boolean_mask/strided_slice:output:0,boolean_mask/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask/Prod]
boolean_mask/Shape_1Shapex*
T0*
_output_shapes
:2
boolean_mask/Shape_1?
"boolean_mask/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask/strided_slice_1/stack?
$boolean_mask/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_1/stack_1?
$boolean_mask/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_1/stack_2?
boolean_mask/strided_slice_1StridedSliceboolean_mask/Shape_1:output:0+boolean_mask/strided_slice_1/stack:output:0-boolean_mask/strided_slice_1/stack_1:output:0-boolean_mask/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2
boolean_mask/strided_slice_1]
boolean_mask/Shape_2Shapex*
T0*
_output_shapes
:2
boolean_mask/Shape_2?
"boolean_mask/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2$
"boolean_mask/strided_slice_2/stack?
$boolean_mask/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask/strided_slice_2/stack_1?
$boolean_mask/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask/strided_slice_2/stack_2?
boolean_mask/strided_slice_2StridedSliceboolean_mask/Shape_2:output:0+boolean_mask/strided_slice_2/stack:output:0-boolean_mask/strided_slice_2/stack_1:output:0-boolean_mask/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2
boolean_mask/strided_slice_2?
boolean_mask/concat/values_1Packboolean_mask/Prod:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat/values_1v
boolean_mask/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/concat/axis?
boolean_mask/concatConcatV2%boolean_mask/strided_slice_1:output:0%boolean_mask/concat/values_1:output:0%boolean_mask/strided_slice_2:output:0!boolean_mask/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask/concat?
boolean_mask/ReshapeReshapexboolean_mask/concat:output:0*
T0*#
_output_shapes
:?????????2
boolean_mask/Reshape?
boolean_mask/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2
boolean_mask/Reshape_1/shape?
boolean_mask/Reshape_1ReshapeAll:output:0%boolean_mask/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2
boolean_mask/Reshape_1{
boolean_mask/WhereWhereboolean_mask/Reshape_1:output:0*'
_output_shapes
:?????????2
boolean_mask/Where?
boolean_mask/SqueezeSqueezeboolean_mask/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2
boolean_mask/Squeezez
boolean_mask/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask/GatherV2/axis?
boolean_mask/GatherV2GatherV2boolean_mask/Reshape:output:0boolean_mask/Squeeze:output:0#boolean_mask/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
boolean_mask/GatherV2]
boolean_mask_1/ShapeShapey*
T0*
_output_shapes
:2
boolean_mask_1/Shape?
"boolean_mask_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2$
"boolean_mask_1/strided_slice/stack?
$boolean_mask_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_1?
$boolean_mask_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice/stack_2?
boolean_mask_1/strided_sliceStridedSliceboolean_mask_1/Shape:output:0+boolean_mask_1/strided_slice/stack:output:0-boolean_mask_1/strided_slice/stack_1:output:0-boolean_mask_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
boolean_mask_1/strided_slice?
%boolean_mask_1/Prod/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB: 2'
%boolean_mask_1/Prod/reduction_indices?
boolean_mask_1/ProdProd%boolean_mask_1/strided_slice:output:0.boolean_mask_1/Prod/reduction_indices:output:0*
T0*
_output_shapes
: 2
boolean_mask_1/Proda
boolean_mask_1/Shape_1Shapey*
T0*
_output_shapes
:2
boolean_mask_1/Shape_1?
$boolean_mask_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$boolean_mask_1/strided_slice_1/stack?
&boolean_mask_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_1/stack_1?
&boolean_mask_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_1/stack_2?
boolean_mask_1/strided_slice_1StridedSliceboolean_mask_1/Shape_1:output:0-boolean_mask_1/strided_slice_1/stack:output:0/boolean_mask_1/strided_slice_1/stack_1:output:0/boolean_mask_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *

begin_mask2 
boolean_mask_1/strided_slice_1a
boolean_mask_1/Shape_2Shapey*
T0*
_output_shapes
:2
boolean_mask_1/Shape_2?
$boolean_mask_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2&
$boolean_mask_1/strided_slice_2/stack?
&boolean_mask_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2(
&boolean_mask_1/strided_slice_2/stack_1?
&boolean_mask_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&boolean_mask_1/strided_slice_2/stack_2?
boolean_mask_1/strided_slice_2StridedSliceboolean_mask_1/Shape_2:output:0-boolean_mask_1/strided_slice_2/stack:output:0/boolean_mask_1/strided_slice_2/stack_1:output:0/boolean_mask_1/strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
end_mask2 
boolean_mask_1/strided_slice_2?
boolean_mask_1/concat/values_1Packboolean_mask_1/Prod:output:0*
N*
T0*
_output_shapes
:2 
boolean_mask_1/concat/values_1z
boolean_mask_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/concat/axis?
boolean_mask_1/concatConcatV2'boolean_mask_1/strided_slice_1:output:0'boolean_mask_1/concat/values_1:output:0'boolean_mask_1/strided_slice_2:output:0#boolean_mask_1/concat/axis:output:0*
N*
T0*
_output_shapes
:2
boolean_mask_1/concat?
boolean_mask_1/ReshapeReshapeyboolean_mask_1/concat:output:0*
T0*#
_output_shapes
:?????????2
boolean_mask_1/Reshape?
boolean_mask_1/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
boolean_mask_1/Reshape_1/shape?
boolean_mask_1/Reshape_1ReshapeAll:output:0'boolean_mask_1/Reshape_1/shape:output:0*
T0
*#
_output_shapes
:?????????2
boolean_mask_1/Reshape_1?
boolean_mask_1/WhereWhere!boolean_mask_1/Reshape_1:output:0*'
_output_shapes
:?????????2
boolean_mask_1/Where?
boolean_mask_1/SqueezeSqueezeboolean_mask_1/Where:index:0*
T0	*#
_output_shapes
:?????????*
squeeze_dims
2
boolean_mask_1/Squeeze~
boolean_mask_1/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
boolean_mask_1/GatherV2/axis?
boolean_mask_1/GatherV2GatherV2boolean_mask_1/Reshape:output:0boolean_mask_1/Squeeze:output:0%boolean_mask_1/GatherV2/axis:output:0*
Taxis0*
Tindices0	*
Tparams0*#
_output_shapes
:?????????2
boolean_mask_1/GatherV2?
=correlation/stddev/variance/covariance/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2?
=correlation/stddev/variance/covariance/Mean/reduction_indices?
+correlation/stddev/variance/covariance/MeanMeanboolean_mask/GatherV2:output:0Fcorrelation/stddev/variance/covariance/Mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(2-
+correlation/stddev/variance/covariance/Mean?
*correlation/stddev/variance/covariance/subSubboolean_mask/GatherV2:output:04correlation/stddev/variance/covariance/Mean:output:0*
T0*#
_output_shapes
:?????????2,
*correlation/stddev/variance/covariance/sub?
*correlation/stddev/variance/covariance/mulMul.correlation/stddev/variance/covariance/sub:z:0.correlation/stddev/variance/covariance/sub:z:0*
T0*#
_output_shapes
:?????????2,
*correlation/stddev/variance/covariance/mul?
?correlation/stddev/variance/covariance/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2A
?correlation/stddev/variance/covariance/Mean_1/reduction_indices?
-correlation/stddev/variance/covariance/Mean_1Mean.correlation/stddev/variance/covariance/mul:z:0Hcorrelation/stddev/variance/covariance/Mean_1/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(2/
-correlation/stddev/variance/covariance/Mean_1?
correlation/stddev/SqrtSqrt6correlation/stddev/variance/covariance/Mean_1:output:0*
T0*
_output_shapes
:2
correlation/stddev/Sqrt?
correlation/truedivRealDivboolean_mask/GatherV2:output:0correlation/stddev/Sqrt:y:0*
T0*#
_output_shapes
:?????????2
correlation/truediv?
?correlation/stddev_1/variance/covariance/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2A
?correlation/stddev_1/variance/covariance/Mean/reduction_indices?
-correlation/stddev_1/variance/covariance/MeanMean boolean_mask_1/GatherV2:output:0Hcorrelation/stddev_1/variance/covariance/Mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(2/
-correlation/stddev_1/variance/covariance/Mean?
,correlation/stddev_1/variance/covariance/subSub boolean_mask_1/GatherV2:output:06correlation/stddev_1/variance/covariance/Mean:output:0*
T0*#
_output_shapes
:?????????2.
,correlation/stddev_1/variance/covariance/sub?
,correlation/stddev_1/variance/covariance/mulMul0correlation/stddev_1/variance/covariance/sub:z:00correlation/stddev_1/variance/covariance/sub:z:0*
T0*#
_output_shapes
:?????????2.
,correlation/stddev_1/variance/covariance/mul?
Acorrelation/stddev_1/variance/covariance/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2C
Acorrelation/stddev_1/variance/covariance/Mean_1/reduction_indices?
/correlation/stddev_1/variance/covariance/Mean_1Mean0correlation/stddev_1/variance/covariance/mul:z:0Jcorrelation/stddev_1/variance/covariance/Mean_1/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(21
/correlation/stddev_1/variance/covariance/Mean_1?
correlation/stddev_1/SqrtSqrt8correlation/stddev_1/variance/covariance/Mean_1:output:0*
T0*
_output_shapes
:2
correlation/stddev_1/Sqrt?
correlation/truediv_1RealDiv boolean_mask_1/GatherV2:output:0correlation/stddev_1/Sqrt:y:0*
T0*#
_output_shapes
:?????????2
correlation/truediv_1?
-correlation/covariance/Mean/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 2/
-correlation/covariance/Mean/reduction_indices?
correlation/covariance/MeanMeancorrelation/truediv:z:06correlation/covariance/Mean/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(2
correlation/covariance/Mean?
correlation/covariance/subSubcorrelation/truediv:z:0$correlation/covariance/Mean:output:0*
T0*#
_output_shapes
:?????????2
correlation/covariance/sub?
/correlation/covariance/Mean_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 21
/correlation/covariance/Mean_1/reduction_indices?
correlation/covariance/Mean_1Meancorrelation/truediv_1:z:08correlation/covariance/Mean_1/reduction_indices:output:0*
T0*
_output_shapes
:*
	keep_dims(2
correlation/covariance/Mean_1?
correlation/covariance/sub_1Subcorrelation/truediv_1:z:0&correlation/covariance/Mean_1:output:0*
T0*#
_output_shapes
:?????????2
correlation/covariance/sub_1?
correlation/covariance/mulMulcorrelation/covariance/sub:z:0 correlation/covariance/sub_1:z:0*
T0*#
_output_shapes
:?????????2
correlation/covariance/mul?
/correlation/covariance/Mean_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B : 21
/correlation/covariance/Mean_2/reduction_indices?
correlation/covariance/Mean_2Meancorrelation/covariance/mul:z:08correlation/covariance/Mean_2/reduction_indices:output:0*
T0*
_output_shapes
: 2
correlation/covariance/Mean_2i
IdentityIdentity&correlation/covariance/Mean_2:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*1
_input_shapes 
:?????????:?????????:F B
#
_output_shapes
:?????????

_user_specified_namex:FB
#
_output_shapes
:?????????

_user_specified_namey
?
j
!__inference_signature_wrapper_239

scores

user_index
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallscores
user_index*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *!
fR
__inference___call___2312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*1
_input_shapes 
:??????????????????: 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????????????
 
_user_specified_namescores:B>

_output_shapes
: 
$
_user_specified_name
user_index
?
E
__inference__traced_restore_273
file_prefix

identity_1??
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:*
dtypes
22
	RestoreV29
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpd
IdentityIdentityfile_prefix^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityX

Identity_1IdentityIdentity:output:0*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*
_input_shapes
: :C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
map_while_body_24$
 map_while_map_while_loop_counter
map_while_map_strided_slice
map_while_placeholder
map_while_placeholder_1#
map_while_map_strided_slice_1_0_
[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0
map_while_reshape_0
map_while_identity
map_while_identity_1
map_while_identity_2
map_while_identity_3!
map_while_map_strided_slice_1]
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor
map_while_reshape?
;map/while/TensorArrayV2Read/TensorListGetItem/element_shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2=
;map/while/TensorArrayV2Read/TensorListGetItem/element_shape?
-map/while/TensorArrayV2Read/TensorListGetItemTensorListGetItem[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0map_while_placeholderDmap/while/TensorArrayV2Read/TensorListGetItem/element_shape:output:0*#
_output_shapes
:?????????*
element_dtype02/
-map/while/TensorArrayV2Read/TensorListGetItem?
map/while/PartitionedCallPartitionedCallmap_while_reshape_04map/while/TensorArrayV2Read/TensorListGetItem:item:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *$
fR
__inference_cor_na_omit_1362
map/while/PartitionedCall?
.map/while/TensorArrayV2Write/TensorListSetItemTensorListSetItemmap_while_placeholder_1map_while_placeholder"map/while/PartitionedCall:output:0*
_output_shapes
: *
element_dtype020
.map/while/TensorArrayV2Write/TensorListSetItemd
map/while/add/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add/yy
map/while/addAddV2map_while_placeholdermap/while/add/y:output:0*
T0*
_output_shapes
: 2
map/while/addh
map/while/add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
map/while/add_1/y?
map/while/add_1AddV2 map_while_map_while_loop_countermap/while/add_1/y:output:0*
T0*
_output_shapes
: 2
map/while/add_1j
map/while/IdentityIdentitymap/while/add_1:z:0*
T0*
_output_shapes
: 2
map/while/Identityv
map/while/Identity_1Identitymap_while_map_strided_slice*
T0*
_output_shapes
: 2
map/while/Identity_1l
map/while/Identity_2Identitymap/while/add:z:0*
T0*
_output_shapes
: 2
map/while/Identity_2?
map/while/Identity_3Identity>map/while/TensorArrayV2Write/TensorListSetItem:output_handle:0*
T0*
_output_shapes
: 2
map/while/Identity_3"1
map_while_identitymap/while/Identity:output:0"5
map_while_identity_1map/while/Identity_1:output:0"5
map_while_identity_2map/while/Identity_2:output:0"5
map_while_identity_3map/while/Identity_3:output:0"@
map_while_map_strided_slice_1map_while_map_strided_slice_1_0"(
map_while_reshapemap_while_reshape_0"?
Ymap_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor[map_while_tensorarrayv2read_tensorlistgetitem_map_tensorarrayunstack_tensorlistfromtensor_0*.
_input_shapes
: : : : : : :?????????: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :)%
#
_output_shapes
:?????????"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
B
scores8
serving_default_scores:0??????????????????
0

user_index"
serving_default_user_index:0 <
output_00
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?
<

signatures
__call__"
_generic_user_object
,
serving_default"
signature_map
?2?
__inference___call___231?
???
FullArgSpec+
args#? 
jself
jscores
j
user_index
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? */?,
!???????????????????
? 
9B7
!__inference_signature_wrapper_239scores
user_index?
__inference___call___231iM?J
C?@
)?&
scores??????????????????
?

user_index 
? "???????????
!__inference_signature_wrapper_239?e?b
? 
[?X
3
scores)?&
scores??????????????????
!

user_index?

user_index "3?0
.
output_0"?
output_0?????????