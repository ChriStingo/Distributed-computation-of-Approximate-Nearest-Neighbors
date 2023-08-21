from os import listdir
import numpy as np


PATH_DATASETS = '../../Datasets/Vectors/' # Path of datasets folder
PATH_INDEX = '../../Datasets/Indexes/knnIndex.knn' # Index name
PATH_IMAGES = '../../Datasets/Images/images.txt'

def DEBUG(elements):
    print(' '.join(map(str, elements)))

MOCKED_QUERY_VECTOR = [[3.37370228767395,1.3559612035751343,0.0,1.3643430471420288,0.9732813835144043,0.0,0.0,0.5503377914428711,1.9339576959609985,10.166431427001953,0.0,5.695069313049316,0.0,1.3337907791137695,2.0418701171875,2.0100655555725098,0.0,0.0,0.16698169708251953,0.0,3.2938997745513916,4.763373374938965,1.5887494087219238,2.5123486518859863,0.0,0.7755516767501831,3.657271385192871,0.0,0.0,4.539532661437988,3.9271631240844727,1.3954927921295166,8.070151329040527,5.056360721588135,0.012419939041137695,1.1468267440795898,0.0,0.0,2.8251514434814453,0.0,1.310355305671692,5.308654308319092,2.4897584915161133,12.043447494506836,1.224100112915039,0.0,3.20660400390625,0.7459313869476318,0.0,2.8149921894073486,0.0,6.292747974395752,0.6774940490722656,9.538518905639648,0.00040268898010253906,1.027884840965271,2.8099160194396973,2.8465232849121094,0.437774658203125,1.697588324546814,4.070858001708984,2.6134142875671387,6.132697105407715,0.0,5.538110256195068,1.0415222644805908,0.0,11.825084686279297,0.4888465404510498,2.982207775115967,9.632806777954102,1.6651372909545898,1.0075304508209229,1.2442035675048828,0.5459640026092529,10.53581428527832,0.0,2.361891984939575,12.621698379516602,0.7354581356048584,9.853328704833984,2.5258541107177734,0.7215427160263062,0.0,3.2446274757385254,10.26927375793457,0.759453535079956,0.0,3.3804073333740234,1.1519153118133545,1.1281709671020508,1.485482931137085,2.146361827850342,1.044437050819397,1.3636689186096191,0.0,6.385831356048584,0.008810043334960938,4.956820487976074,0.04658317565917969,3.1830081939697266,6.743096828460693,0.629848837852478,0.0,0.24547481536865234,0.0,0.0,0.0,0.0,16.073883056640625,2.0006165504455566,0.0,0.18894624710083008,9.992981910705566,1.3323147296905518,0.12587499618530273,1.003254771232605,2.001838207244873,0.20627927780151367,0.7287725210189819,2.2585549354553223,7.928247451782227,13.884439468383789,8.819491386413574,4.879702568054199,1.5427731275558472,1.9763209819793701,10.491185188293457,0.0,6.429313659667969,3.4517083168029785,1.6643424034118652,0.0,0.0,5.696453094482422,0.0,6.9279985427856445,2.5888493061065674,18.644672393798828,0.7463071346282959,0.0,1.2141022682189941,5.688807487487793,7.840796947479248,1.4061994552612305,0.0,6.745924949645996,0.3967125415802002,3.4222676753997803,0.0,8.53191089630127,2.3602042198181152,1.946789026260376,4.008541107177734,7.049567222595215,0.0,1.6665103435516357,0.0,0.3995920419692993,8.11866569519043,0.43852198123931885,2.229963779449463,3.25439453125,0.0,0.0,16.699766159057617,0.0,4.287610054016113,0.32838869094848633,5.782001495361328,0.9290180206298828,0.0,0.0,0.5224175453186035,0.0,5.59341287612915,0.6491057872772217,0.0,0.9855422973632812,1.1047202348709106,1.2874603271484375e-05,11.06383228302002,0.0,4.8452935218811035,1.2053431272506714,2.0312414169311523,0.0,0.0,0.0,3.3177192211151123,0.4298059940338135,2.072679042816162,1.5969904661178589,2.0923757553100586,0.0,0.16791647672653198,9.260540962219238,2.7597508430480957,0.0,3.5190131664276123,0.0,0.0,9.035099029541016,0.514559268951416,0.0,0.3414804935455322,0.31618475914001465,7.571516036987305,1.318920612335205,4.506561756134033,4.442713737487793,0.0,0.0,0.9146730899810791,0.8316385746002197,0.0,1.2357070446014404,1.2686915397644043,0.24042069911956787,0.0,12.416106224060059,1.2439897060394287,0.9321268796920776,0.0,0.5276155471801758,0.17042016983032227,3.782344341278076,3.5516233444213867,6.876280307769775,0.9164559841156006,0.19007527828216553,2.954557418823242,0.8006279468536377,0.7620662450790405,4.321337699890137,1.8348981142044067,0.6252118349075317,0.0,0.0,1.0370665788650513,0.0,3.367138385772705,0.0,5.731408596038818,2.258817672729492,4.670315742492676,0.0,0.0,2.5413870811462402,3.7559900283813477,0.7151923179626465,0.0,0.0,1.9759159088134766,0.0,0.5863674879074097,2.648254871368408,1.4336599111557007,1.7855641841888428,0.0,0.0,1.1785297393798828,20.43645477294922,0.0,2.9061789512634277,0.8477545976638794,0.12561631202697754,0.0,24.990537643432617,4.091487407684326,0.0,0.0,0.0,0.0,1.1203453540802002,3.309091091156006,0.0,0.0,6.201620101928711,1.2167026996612549,1.675504446029663,0.5213052034378052,0.8357335329055786,0.0,3.5573227405548096,0.28341448307037354,0.7432923316955566,4.930508613586426,12.180251121520996,0.06637430191040039,0.2680257558822632,0.0,0.8878333568572998,1.291363000869751,0.0,5.401802062988281,4.650834560394287,0.2750624418258667,0.7273145914077759,4.539768218994141,0.0,2.1133995056152344,0.6888871192932129,0.0,1.5082377195358276,2.156252145767212,0.49196958541870117,1.3824520111083984,0.8638107776641846,0.594136118888855,0.0,0.6685335636138916,0.0,0.8359341621398926,1.1147254705429077,1.9162368774414062,0.6979870796203613,1.2586289644241333,5.006922721862793,4.749630928039551,4.7308244705200195,0.0,0.0,3.5665810108184814,1.561356782913208,2.652130603790283,0.3630838394165039,2.9052226543426514,0.44408726692199707,6.198277473449707,0.0,0.590766191482544,7.700359344482422,0.06301259994506836,3.3160431385040283,1.8882269859313965,2.911052703857422,0.0,11.170553207397461,0.6812242269515991,1.9307061433792114,5.783306121826172,0.0,0.0,2.6275205612182617,0.0,0.0,0.09798431396484375,1.9086860418319702,7.6427321434021,3.8062021732330322,0.1668691635131836,0.0,5.898550033569336,3.2516191005706787,1.9275914430618286,0.4538700580596924,7.132201194763184,0.0,0.0,2.6821787357330322,5.748713493347168,0.8179726600646973,0.0,0.12556767463684082,1.6531729698181152,5.342517375946045,3.86063289642334,2.1724841594696045,0.6751902103424072,0.0,2.8209497928619385,0.6526541709899902,0.0,0.7795335054397583,0.0,0.0,0.0,17.885379791259766,10.172041893005371,0.0,14.955982208251953,0.0,0.0,3.0639405250549316,0.0,0.8157176971435547,0.0,0.0,0.48963046073913574,0.0,0.8897974491119385,0.346663236618042,0.0,1.6450347900390625,4.9171319007873535,0.0,0.9819308519363403,2.0258231163024902,0.0,0.7020125389099121,1.9940263032913208,2.6828863620758057,0.0,0.0,2.7642300128936768,2.5132224559783936,1.7701945304870605,0.0,14.422715187072754,4.103070259094238,0.9668319225311279,0.9585213661193848,0.0,15.772632598876953,1.732544183731079,0.8080756664276123,8.555614471435547,2.9719488620758057,0.3563448190689087,0.0,4.180280685424805,0.6184349060058594,2.958542585372925,0.0,1.653483271598816,21.27911376953125,0.0,1.2162585258483887,2.9707837104797363,12.769461631774902,0.4379885196685791,20.901269912719727,0.5034985542297363,0.0,4.701305866241455,10.188374519348145,0.0,0.31246232986450195,0.7940628528594971,3.7567267417907715,0.7360575199127197,0.0,9.320564270019531,0.4876701831817627,1.3871148824691772,4.491086006164551,4.754913330078125,2.487980365753174,4.088196754455566,17.73358726501465,0.8503754138946533,1.2830555438995361,6.894021511077881,0.0,10.854588508605957,0.8840423822402954,4.573430061340332,9.899947166442871,0.0,0.7948853969573975,0.827267050743103,9.386102676391602,0.0,0.0,0.05330610275268555,2.6812586784362793,11.072349548339844,2.1114230155944824,5.747990131378174,0.6879817247390747,3.3324577808380127,0.21721792221069336,1.6465387344360352,2.456923007965088,10.041268348693848,5.0765275955200195,6.709954261779785,1.2878695726394653,4.858698844909668,0.0,1.7368919849395752,2.56278133392334,4.426736831665039,0.0,0.0,2.416733741760254,4.022677421569824,2.3762097358703613,5.245064735412598,0.10921478271484375,3.257091999053955,5.233511924743652,1.3829466104507446,5.333559513092041,0.9501445293426514,3.194176197052002,1.9107881784439087,2.0205793380737305,0.0,0.2533721923828125,5.05860710144043,3.569148540496826,0.14521074295043945,9.252549171447754,0.0,3.105818033218384,1.4984042644500732,1.5062212944030762,0.0,14.760339736938477,0.7500364780426025,8.735013961791992,4.548121452331543,0.0,1.8109285831451416,1.0313266515731812,2.1610584259033203,0.5692410469055176,2.9974989891052246,10.048561096191406,4.890448093414307,1.0994672775268555,0.0,3.1293082237243652,11.193641662597656,0.0,12.4236478805542,0.7349369525909424,1.1632747650146484,0.0,1.225369930267334,0.1323108673095703,0.541610598564148,11.199772834777832,9.662678718566895,5.260739326477051,0.0,5.941433429718018,0.0,0.9749792814254761,1.152071237564087,9.583353042602539,0.0,0.0,0.9140920639038086,4.5965776443481445,0.7601807117462158,1.1176965236663818,25.68712615966797,0.7901327610015869,0.0,5.106159210205078,0.06879323720932007,5.402432441711426,0.0,6.793320178985596,1.369149923324585,1.3060555458068848,0.17030298709869385,0.6664518117904663,5.508236885070801,0.0,0.0,0.0,1.4254539012908936,1.401970386505127,0.20416641235351562,0.9354262351989746,4.267902374267578,3.910860776901245,0.31826984882354736,0.18621361255645752,0.0,2.3790814876556396,0.0,0.7572880983352661,0.8068159222602844,5.922468662261963,5.492539882659912,4.5916428565979,6.576871395111084,14.619428634643555,2.9427716732025146,0.6310567855834961,1.903685450553894,0.25582826137542725,1.4675054550170898,3.234151840209961,3.0567433834075928,2.5681605339050293,5.392764091491699,0.0,0.0,2.4078848361968994,0.0,0.5237382650375366,3.5640926361083984,0.8864858150482178,1.9576529264450073,0.0,0.0,0.35369443893432617,4.2382097244262695,8.208712577819824,0.0,0.0,0.0,0.1532604694366455,1.7897282838821411,5.23961067199707,1.9287421703338623,1.7497773170471191,0.48764562606811523,12.763446807861328,3.9237465858459473,3.559781551361084,6.098149299621582,1.2614264488220215,3.410799980163574,2.7814478874206543,1.037665843963623,1.9792628288269043,1.2511522769927979,2.3305931091308594,1.4396690130233765,1.9357866048812866,11.910917282104492,1.6785253286361694,0.0,0.13844585418701172,1.6506900787353516,2.6351265907287598,3.312227249145508,0.0,2.03208065032959,0.4809991121292114,4.51369571685791,8.442580223083496,4.932590961456299,3.972883701324463,1.818934440612793,3.9708971977233887,0.0,4.231486797332764,3.592301368713379,0.48636817932128906,1.3609169721603394,0.0,4.454323768615723,0.0,4.704179763793945,2.384693145751953,6.821155071258545,4.184153079986572,0.0,1.443792462348938,2.186758518218994,0.0,5.097624778747559,7.374750137329102,0.0,3.204977512359619,1.0730626583099365,2.0205063819885254,0.4915781021118164,9.32956314086914,1.5872788429260254,0.0,0.0,0.0,3.9357120990753174,0.0,0.0,2.3881425857543945,5.165863990783691,0.8161096572875977,0.8684803247451782,4.868122577667236,8.904838562011719,0.16582465171813965,4.765030384063721,1.145674705505371,0.0,3.2746615409851074,0.06619644165039062,17.545991897583008,0.0,2.8356094360351562,7.668486595153809,1.2486048936843872,1.2593209743499756,0.0032896995544433594,0.0,0.5112459659576416,0.5860483646392822,0.3595454692840576,1.1590569019317627,0.0,4.087472915649414,12.136062622070312,3.152313232421875,0.0,0.6657040119171143,0.0,0.7991825342178345,5.589395046234131,1.965928077697754,4.1354594230651855,0.41246116161346436,1.124880313873291,0.7421518564224243,0.8766615390777588,0.06756949424743652,1.8517986536026,0.5696835517883301,2.4144039154052734,0.1726452112197876,3.1124470233917236,5.785409927368164,6.7031354904174805,1.8732398748397827,0.11831426620483398,0.0,1.189042091369629,0.0,0.0,1.2112374305725098,8.247116088867188,0.20702672004699707,0.0,2.5953621864318848,5.67291784286499,14.124824523925781,0.0,2.3523004055023193,2.3765625953674316,0.0,1.6909013986587524,0.0,2.631739854812622,2.982239246368408,0.0,0.875943660736084,5.232003211975098,0.0,6.053163051605225,2.7123794555664062,0.34512364864349365,9.299488067626953,1.424820899963379,0.0,1.7974677085876465,1.602559208869934,1.2153756618499756,1.4493255615234375,0.0,0.7977395057678223,10.635069847106934,5.023969650268555,0.0,5.16210412979126,1.4021004438400269,0.0,2.903419017791748,3.6062402725219727,0.6165328025817871,2.2690558433532715,0.0027512311935424805,1.426417589187622,2.967597484588623,0.39163923263549805,1.2152659893035889,0.0,1.340848684310913,0.0,2.625074863433838,7.056306838989258,6.598606109619141,1.0617974996566772,0.30486929416656494,0.0,6.434800148010254,0.0,2.091172695159912,0.0,8.469951629638672,0.5083544254302979,0.055866241455078125,0.9356411695480347,1.5703356266021729,1.6583077907562256,11.844706535339355,0.0,5.41322135925293,10.402182579040527,7.571510314941406,0.0,0.0,0.0,2.3742706775665283,1.1419150829315186,0.5775483846664429,2.4967379570007324,4.866827964782715,4.746354579925537,3.8198001384735107,10.647969245910645,0.2485262155532837,0.0,0.0,3.6384682655334473,2.440732479095459,4.2792768478393555,0.0,0.0,1.6168842315673828,1.9441231489181519,1.5903511047363281,0.06205940246582031,0.0,3.775714635848999,0.16240262985229492,9.002147674560547,22.14934730529785,0.758033037185669,0.8785684108734131,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,3.210512399673462,0.0,0.0,0.0,3.257547378540039,1.1849877834320068,18.5084171295166,3.3223326206207275,0.29413342475891113,4.882632732391357,8.738003730773926,0.09661567211151123,0.0,2.1735310554504395,4.703063011169434,0.0,1.2180366516113281,2.7217936515808105,0.0,2.0855422019958496,0.949599027633667,1.2406731843948364,0.22109770774841309,0.6381274461746216,0.6845152378082275,0.0,0.0,4.622439384460449,2.180410861968994,2.854417324066162,1.843566656112671,0.0,0.3853803873062134,2.2755179405212402,3.8669066429138184,0.0,0.5249137878417969,0.0,5.640514850616455,0.0,0.0,3.1491270065307617,3.8542842864990234,0.10978734493255615,0.11112701892852783,0.0,0.0,1.791391372680664,2.699476718902588,1.6634893417358398,0.1685013771057129,0.5147421360015869,4.690889358520508,3.185169219970703,2.250727653503418,0.35490089654922485,0.0,6.504593372344971,2.785656213760376,9.215434074401855,0.0,6.491829872131348,1.6813675165176392,15.879266738891602,2.83809757232666,4.916358470916748,4.877542495727539,0.6833460330963135,1.942355751991272,0.0,0.3034336566925049,0.0,3.0175466537475586,0.0,0.0,0.8308385610580444,0.0,0.0,1.1178555488586426,5.220820426940918,1.0774368047714233,0.25121521949768066,0.0,1.399600625038147,0.0,0.0,0.3674502372741699,2.2405364513397217,7.9076714515686035,0.0,0.0,2.20938777923584,0.3673572540283203,1.3974597454071045,0.0,0.7445131540298462,0.0,2.041734457015991,0.0,1.5872169733047485,2.195216178894043,0.8602521419525146,3.0160210132598877,8.905731201171875,0.4907381534576416,0.40767812728881836,1.2915832996368408,1.0541809797286987,3.4043593406677246,3.671408176422119,0.0,4.750727653503418,1.9138237237930298,0.9001778364181519,3.306325674057007,9.961396217346191,0.0,0.3635441064834595,0.2568246126174927,0.0,7.8502349853515625,0.0,1.0785973072052002,1.9665074348449707,0.0,11.216148376464844,0.0,3.524780511856079,0.0,2.5038747787475586,1.3643953800201416,0.0,10.075845718383789,3.273231029510498,0.0,3.7356247901916504,3.421689033508301,0.3601055145263672,0.0,0.0,0.4692575931549072,15.426935195922852,5.219044208526611,4.3601393699646,0.0,1.2106149196624756,1.7899549007415771,0.0,2.489149808883667,0.40830016136169434,1.020862340927124,6.867242336273193,4.420952796936035,0.02828192710876465,1.8538830280303955,2.2973527908325195,0.0,6.615620136260986,4.145443916320801,3.7302846908569336,0.0,0.0,3.659729242324829,0.0,0.0,4.6588850021362305,0.60679030418396,1.6332405805587769,3.6333255767822266,3.864896774291992,0.0,4.081233501434326,3.036064624786377,4.629770278930664,2.722609758377075,0.9065341949462891,0.5549454689025879,0.28285980224609375,0.8353776931762695,21.57764434814453,1.034839153289795,0.0,0.789272665977478,0.0,0.4169656038284302,2.3589468002319336,1.7248772382736206,0.0036312341690063477,1.4895901679992676,0.8507169485092163,0.0,0.7591111660003662,0.05308175086975098,2.2672996520996094,0.0,0.7138649225234985,3.7409121990203857,16.123342514038086,0.0,0.044014930725097656,5.2157769203186035,0.0,0.743018388748169,0.0,1.2078872919082642,4.918666839599609,4.285252094268799,2.0117478370666504,0.0,2.3315370082855225,0.0,0.0,5.151556491851807,0.0,0.0,0.0,4.327296257019043,2.851789951324463,0.0,1.2760725021362305,0.0,1.1976393461227417,2.6281259059906006,0.38466525077819824,0.8362773656845093,1.450620174407959,21.66842269897461,0.11242902278900146,0.0,0.0,0.0712118148803711,1.6188547611236572,1.9649696350097656,1.598297119140625,0.6602026224136353,4.073328018188477,2.3882246017456055,6.323868751525879,0.0,1.8462059497833252,2.9837145805358887,2.0390329360961914,0.16006159782409668,1.6795451641082764,2.0778045654296875,0.0,0.0,3.867220878601074,0.0,0.0,1.2617167234420776,1.2320085763931274,0.0,4.009361267089844,2.526604175567627,0.05914568901062012,0.778495192527771,1.1148433685302734,5.731555938720703,2.3062620162963867,2.083012342453003,0.0,0.14702296257019043,4.787385940551758,5.822726726531982,1.9486840963363647,0.0,0.0,0.05803501605987549,1.8580796718597412,3.305612087249756,0.0,2.8273367881774902,5.243649005889893,0.26891255378723145,0.0,1.4215657711029053,5.20680046081543,1.0957581996917725,3.488240957260132,0.0,3.0931496620178223,1.5813429355621338,4.6513776779174805,0.0,0.1488257646560669,0.0,1.4922304153442383,4.88698673248291,1.099365234375,0.6065690517425537,0.1479838490486145,0.07838869094848633,1.1930229663848877,2.783585786819458,1.1511861085891724,1.1111750602722168,0.0,0.0,1.152175784111023,1.5144575834274292,3.126008987426758,15.33413314819336,0.21956515312194824,0.0,1.9760788679122925,1.598052978515625,3.0997378826141357,6.0103864669799805,0.0,0.0,2.657379627227783,0.0,2.533599853515625,10.688928604125977,23.114152908325195,2.3211231231689453,5.480967998504639,1.2371307611465454,1.0623008012771606,0.0,5.669122219085693,0.43993568420410156,1.984681248664856,0.0,0.0,1.2515349388122559,3.1694703102111816,2.2302331924438477,3.3146884441375732,0.0,5.661581993103027,1.6533169746398926,1.9432412385940552,3.676886558532715,0.7629108428955078,3.880974531173706,0.0,1.9260368347167969,0.0,0.25883400440216064,11.020030975341797,2.097658634185791,0.31658434867858887,2.806290626525879,0.0,3.08528995513916,3.1431634426116943,0.0,0.0,0.4445209503173828,2.6622676849365234,1.6983866691589355,6.555011749267578,0.0,4.540426731109619,0.0,2.7359890937805176,0.0,0.0,5.3235626220703125,0.0,2.66363263130188,0.0,2.37612247467041,2.4435746669769287,0.012096524238586426,3.7473044395446777,0.0,0.4154391288757324,0.6146845817565918,2.386038303375244,1.095870852470398,0.0,0.5818592309951782,2.635951519012451,1.2148895263671875,0.0,0.0,3.1729321479797363,1.5468544960021973,0.5308759212493896,0.1766291856765747,1.8815113306045532,0.0,2.5392587184906006,3.5525386333465576,16.004230499267578,1.9639815092086792,0.17169439792633057,2.985395908355713,0.0,15.011627197265625,1.0326184034347534,0.0,4.789243221282959,0.49155592918395996,2.1441521644592285,4.164938449859619,0.0,5.871798515319824,2.37038516998291,0.0,4.575945854187012,1.4351776838302612,7.714824676513672,0.6191649436950684,0.0,8.830084800720215,2.179960250854492,0.0,0.0,0.0,1.5806069374084473,0.0,5.993956565856934,0.0,8.009895324707031,1.43152916431427,0.0,0.0,0.8726164102554321,2.933593273162842,0.34174466133117676,6.574033260345459,0.14933180809020996,3.453054428100586,0.0,6.7642059326171875,1.7860636711120605,6.8698225021362305,0.0,3.111222982406616,1.8832616806030273,0.0,0.3756539821624756,0.0,3.1902079582214355,0.0,0.0,5.416553497314453,0.04317176342010498,0.209969162940979,11.264395713806152,0.0,10.909242630004883,0.0,0.3918423652648926,2.135040283203125,1.2313052415847778,5.386836051940918,0.18123412132263184,1.723015308380127,2.1990561485290527,3.1950628757476807,5.393779277801514,2.2152044773101807,6.478627681732178,2.998821258544922,2.4837262630462646,0.0,1.3786484003067017,0.6477593183517456,0.4912114143371582,0.972286581993103,0.0,0.9031813144683838,2.4975357055664062,0.0,1.2666735649108887,0.0,0.0,0.1666480302810669,0.8407357931137085,0.8121856451034546,2.8309662342071533,3.0192441940307617,2.477147340774536,0.0,0.0,2.411501884460449,0.0,2.0500130653381348,0.0,7.420309543609619,0.0,1.6732152700424194,3.8688158988952637,0.06310462951660156,0.0,3.780357837677002,0.0,0.0,2.243250846862793,0.0,0.0,0.0,1.3193254470825195,2.3038902282714844,0.0,5.579638957977295,1.7072863578796387,6.195187568664551,0.0,1.9277749061584473,0.0,2.031873941421509,6.827037334442139,18.099868774414062,1.7388049364089966,4.2546892166137695,3.025540590286255,3.489569664001465,1.6703919172286987,1.8157752752304077,1.0868905782699585,1.5732769966125488,0.329121470451355,1.4424455165863037,0.0,3.1950974464416504,0.21764624118804932,14.893200874328613,0.20092248916625977,15.495068550109863,0.0,3.8289084434509277,0.0,2.681631565093994,0.0,0.6908886432647705,0.0,0.0,1.212340235710144,0.0,1.0950860977172852,0.6224979162216187,1.6250571012496948,0.0,0.7212352752685547,7.244416236877441,5.913647651672363,0.0,8.243857383728027,1.690596103668213,0.0,0.7329189777374268,0.0,1.1310513019561768,0.0,0.0,0.19531893730163574,4.786405563354492,0.0,0.0,2.9961776733398438,0.0,0.980947732925415,12.877676963806152,5.438022613525391,1.0872085094451904,0.0,0.9948527812957764,0.0,0.0,2.5309367179870605,3.598003625869751,0.0,0.745976448059082,0.7136993408203125,9.426239967346191,11.135931015014648,0.0,1.2432599067687988,15.362768173217773,0.4057505130767822,1.2991218566894531,11.72878646850586,1.6993470191955566,0.2728177309036255,0.9038723707199097,2.8747785091400146,0.0,0.0,13.186151504516602,0.1036684513092041,1.644270896911621,5.669247627258301,0.38906168937683105,6.294092655181885,4.09991455078125,0.3908827304840088,4.368952751159668,4.447558403015137,1.2112209796905518,1.521615743637085,0.0,1.2541460990905762,5.041008949279785,2.352893352508545,25.73297119140625,1.230334997177124,0.0,5.214085578918457,0.0,2.486717700958252,1.404314637184143,0.0,0.403109073638916,0.0,2.506875514984131,0.8427610397338867,5.676047325134277,0.0,1.5422842502593994,0.0,0.34987401962280273,3.9192891120910645,0.3877072334289551,13.421663284301758,5.96205472946167,1.6309704780578613,6.436022758483887,3.5607380867004395,5.5823974609375,0.4948890209197998,5.858830451965332,0.08726143836975098,0.0,0.0,3.696079969406128,0.0,0.0,6.600040435791016,2.0398635864257812,2.1207728385925293,5.851816177368164,3.0363783836364746,0.0,0.8004634380340576,1.2751059532165527,7.049015045166016,0.0,0.76297926902771,0.9785773754119873,2.334348678588867,4.935477256774902,1.05501127243042,0.0,0.0,16.148996353149414,0.4828351140022278,0.0,1.3221616744995117,15.076932907104492,0.29116499423980713,1.503389596939087,1.2271147966384888,0.0,0.2658202648162842,0.0,4.281124114990234,0.0,3.446566581726074,4.873125076293945,11.107646942138672,6.061261177062988,2.0821900367736816,11.188199043273926,1.3105441331863403,0.0,0.0,2.842294692993164,7.996847152709961,0.0,0.6454358100891113,8.68963623046875,6.950508117675781,2.113210916519165,2.071059465408325,3.698381185531616,5.731567859649658,4.780124664306641,2.9336488246917725,0.25027918815612793,1.8831336498260498,0.0,3.5233347415924072,2.125964641571045,0.0,2.5405941009521484,6.2071051597595215,0.0,0.5621103048324585,4.028765678405762,4.497269630432129,0.6712498664855957,1.7223931550979614,12.385732650756836,0.1544935703277588,0.0,0.0,3.6530814170837402,1.488403558731079,0.0,0.583794355392456,0.0,0.0,4.279077529907227,0.0,0.0,1.8875226974487305,0.16399800777435303,1.3125816583633423,0.0,7.524909973144531,0.0,1.9499943256378174,4.49323844909668,2.2796006202697754,6.466007232666016,2.827190399169922,1.7995741367340088,0.0,0.20569396018981934,8.027946472167969,0.0,1.8432745933532715,1.4603289365768433,1.6208007335662842,7.691911220550537,4.844996452331543,4.593396186828613,5.914238929748535,0.0,8.89052963256836,0.46867918968200684,2.1152002811431885,0.0,0.0,2.6637015342712402,8.172605514526367,0.0,2.21972918510437,9.045724868774414,0.1872175931930542,0.0,4.789295673370361,0.0,3.9958884716033936,0.0,1.8447046279907227,5.577775001525879,2.822957992553711,1.4585387706756592,1.5637056827545166,2.300318956375122,0.38494133949279785,0.0,2.872462272644043,0.0,1.4952454566955566,12.618426322937012,1.7153087854385376,0.0,0.5023555755615234,0.0,0.7625319957733154,0.5255692005157471,0.0,0.0,0.0,2.9122729301452637,3.241539239883423,1.8519947528839111,2.709190845489502,0.6612952947616577,3.4172120094299316,0.18745815753936768,0.0,11.382758140563965,2.7329764366149902,4.095641613006592,0.0,0.0,5.714201927185059,2.2092983722686768,4.81689453125,7.620977401733398,9.940980911254883,0.0,3.6927759647369385,0.0,7.021175384521484,0.2123568058013916,2.7229199409484863,0.1850430965423584,3.7056643962860107,3.9510598182678223,5.452793598175049,2.1127734184265137,0.0,3.601774215698242,0.0495913028717041,0.0,5.428702354431152,0.5904474258422852,1.6495239734649658,0.764670729637146,0.20518743991851807,3.8039042949676514,0.6568717956542969,0.0,0.7343454360961914,1.417224407196045,0.0,9.048715591430664,3.0637669563293457,1.0618369579315186,0.1486668586730957,7.814262390136719,1.1470032930374146,5.289244651794434,0.0,0.5387704372406006,0.0,0.0,0.7544429302215576,0.0,0.03206944465637207,9.641077995300293,13.46003532409668,0.0,5.564487934112549,5.468859672546387,21.968746185302734,2.7736544609069824,0.3887377977371216,0.8284037113189697,1.7028568983078003,2.1210532188415527,0.3000490665435791,0.0036089420318603516,0.39055967330932617,8.695606231689453,3.289872169494629,2.129659414291382,0.0,0.0,0.9482307434082031,0.13223469257354736,0.06367617845535278,5.76141357421875,0.9270662069320679,1.8237407207489014,0.8327559232711792,2.687737464904785,3.969724655151367,0.6521368026733398,5.269018173217773,0.0,0.6836369037628174,1.8686914443969727,0.0,0.0,2.0284576416015625,2.658639907836914,2.6269235610961914,6.290871620178223,0.05300402641296387,0.0,0.30266791582107544,0.3387889862060547,9.657669067382812,0.0,18.42724609375,0.1723034381866455,7.961542129516602,3.1416754722595215,3.0673177242279053,0.0,0.7173252105712891,1.6144380569458008,0.0,3.699932098388672,1.2053049802780151,1.6063566207885742,0.039421677589416504,0.0,4.380388259887695,0.5248825550079346,15.059429168701172,1.8562300205230713,0.0,0.0,5.602177143096924,0.27777767181396484,0.9742374420166016,0.5084543228149414,1.0946018695831299,1.7008466720581055,2.431184768676758,0.0,0.0,0.0,0.33390605449676514,4.717333793640137,0.0,0.0,6.065803527832031,5.74208402633667,0.19033372402191162,0.22739887237548828,0.45824575424194336,0.0,0.0,17.009708404541016,3.6270742416381836,0.06879818439483643,0.008279919624328613,5.275018215179443,12.492988586425781,2.2123379707336426,6.0927734375,1.4856250286102295,0.9557886123657227,0.0,14.937950134277344,0.0,6.57668399810791,3.331644296646118,0.0,1.816801905632019,3.844472885131836,3.822453498840332,3.2756333351135254,2.6179187297821045,0.38634777069091797,1.332235336303711,7.744311332702637,1.6727238893508911,3.7224273681640625,1.535780668258667,6.42504358291626,0.020385026931762695,9.536149978637695,0.3001737594604492,2.532578945159912,0.0,0.0,3.449672222137451,0.20402216911315918,7.0503997802734375,0.7848696708679199,5.222604751586914,1.2305045127868652,0.8422284126281738,0.0,0.0,0.6287583112716675,0.26299262046813965,1.2452797889709473,1.9134531021118164,0.0,6.873159408569336,6.95533561706543,1.4924887418746948,4.807193279266357,11.044666290283203,3.5198028087615967,0.730165958404541,3.5529625415802,0.8537969589233398,2.462697982788086,0.0,1.0698843002319336,3.866708993911743,3.890162944793701,1.0586886405944824,0.8936048746109009,0.0,3.564497947692871,1.329435110092163,4.929527282714844,0.0,0.0,4.9078497886657715,4.6107258796691895,0.0,3.7358994483947754,0.0,0.0,0.8751277923583984,1.3490768671035767,0.646259069442749,0.38657617568969727,0.0,0.0,4.305737495422363,0.0,0.24106824398040771,0.0,3.9038305282592773,0.0,0.0,9.315322875976562,0.0,9.146788597106934,6.765415668487549,1.7163166999816895,0.9316548109054565,8.721320152282715,0.0,4.356001853942871,5.322848796844482,2.023751974105835,0.0,2.512765407562256,3.53823184967041,0.0,0.0,0.0,15.849894523620605,0.0,0.8666141033172607,2.4213151931762695,0.0,0.0,0.2124636173248291,5.342975616455078,2.4563088417053223,0.418074369430542,4.074487686157227,5.8878021240234375,6.331441402435303,0.675788164138794,0.0,3.358869791030884,2.4938037395477295,0.0,0.0,2.7848405838012695,2.0690932273864746,2.8914427757263184,0.0,0.22828316688537598,0.8191006183624268,1.1940667629241943,0.0,7.640105247497559,0.0,0.8311169147491455,0.0,22.298206329345703,4.777149200439453,1.0205769538879395,6.006817817687988,3.9314193725585938,0.0,7.410484313964844,2.2757320404052734,6.029073715209961,2.010054588317871,0.7658874988555908,1.7231109142303467,1.3563363552093506,0.0,1.6740862131118774,0.7162457704544067,2.5997986793518066,3.622570037841797,0.0,3.924455165863037,0.19505834579467773,7.120895862579346,2.1926708221435547,0.0,8.066898345947266,0.0,0.9211134910583496,1.4542686939239502,0.0,8.84131908416748,0.0,0.8780698776245117,1.5392099618911743,6.306838035583496,8.92335319519043,0.8102078437805176,17.75442886352539,1.9170660972595215,0.0,0.0,2.5458250045776367,0.034774065017700195,9.92259407043457,7.076446533203125,0.15411150455474854,6.712442398071289,0.4273953437805176,0.8048343658447266,3.169158935546875,18.579565048217773,1.8414289951324463,3.32320499420166,0.56253981590271,9.725000381469727,0.2975722551345825,0.0,1.763630986213684,0.41644489765167236,0.0,0.0,0.9800541400909424,0.06853294372558594,8.588102340698242,5.408254623413086,1.880910873413086,0.0,9.873685836791992,7.505120277404785,0.8237671852111816,5.769228458404541,0.0,3.3270387649536133,0.3835742473602295,0.07226181030273438,3.030001163482666,9.746091842651367,1.0695892572402954,2.9867935180664062,1.7360037565231323,0.0,2.323228120803833,2.550645351409912,14.825571060180664,2.518369674682617,1.3889164924621582,3.4817211627960205,0.29033493995666504,0.0,0.48117589950561523,2.496487855911255,2.135178804397583,0.0,0.7503564357757568,0.0,0.0,3.775515079498291,2.050187826156616,4.956279754638672,0.0,3.4075419902801514,0.2719457149505615,0.0,3.5141193866729736,0.33622944355010986,0.0,4.99348258972168,1.546283483505249,23.887537002563477,2.8944203853607178,3.126298427581787,0.0,0.04750514030456543,1.351790189743042,1.45449697971344,1.5744973421096802,0.0,2.4259653091430664,0.013307332992553711,1.0476560592651367,0.0,0.0,1.8757224082946777,2.044124126434326,3.8580217361450195,0.0,1.9588813781738281,2.5187087059020996,0.5805878639221191,1.4343147277832031,3.131319046020508,4.466383457183838,11.955902099609375,0.4295605421066284,0.0,1.7561888694763184,0.0,5.462082862854004,1.4993157386779785,4.308584213256836,3.5739810466766357,0.0,0.3180943727493286,4.699642181396484,0.0,2.8212621212005615,0.0,6.0033159255981445,2.721008062362671,6.0493621826171875,1.6841530799865723,0.0,4.301778793334961,2.63596773147583,0.0,0.34156733751296997,1.9730501174926758,4.681169509887695,3.354339361190796,0.0,1.4947030544281006]]
