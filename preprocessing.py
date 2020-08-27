import pandas as pd

"""
Cleaning and manipulating raw data. Fixing errors, missing values etc.
Replacing values, Marking unknowns of each column
Creating dummies for variables with relevant amount of observations
"""


## Basic Exploration of the Dataset

def describe_raw_data():
    """
    Reading the excel file and printing basic information:
    - Column names, dimension of the dataframe, all variables, amount of observations and type of data per column, basic statistics of each feature
    - Missing Values' Exploration and Feature engeneering
    """
    got_raw = pd.read_excel(r'datafiles/GOT_character_predictions.xlsx')
    print(got_raw.columns)
    print("\n", got_raw.shape)
    print("\n", got_raw.info())
    print("\n", got_raw.describe().round(2))
    print("\n", got_raw.isnull().sum())
    return got_raw

describe_raw_data()

## Dropping categorical variables with more than 80% missing values

def clean_variables(datafile=r'datafiles/GOT_character_predictions.xlsx'):
    """
    missing values treatment, replacing values, dummy treatment
    :return: cleaned df, ready for model building
    """
    got_raw = pd.read_excel(datafile)
    got_raw.drop(["mother", "father", "heir", "spouse", "isAliveMother", "isAliveFather", "isAliveHeir",
                  "isAliveSpouse", "S.No", "age", "dateOfBirth"], axis = 1, inplace = True)


    # Looking at the remaining columns and its missing values
    print(got_raw.isnull().sum())


    #############################################################################
    ## Variable: culture
    # Creating category "Unknown" for missing values

    got_raw["culture"].fillna('Cult_U', inplace = True)

    # Correcting spelling mistakes and naming the same cultures in the same way
    cult_dict = {"Andal":["Andals"], "Asshai": ["Asshai'i"], "Astapor": ["Astapori"], "Braavos": ["Braavosi"], "Dornish": ["Dornishmen", "Dorne"],
                  "Free_Folk":["free folk", "Free folk", "free Folk", "Free folk", "Wildling", "Wildlings", "First men"], "Ghiscari":["Ghiscaricari"], "Ironborn":["Ironmen", "ironborn"], "Lhazareen":["Lhazarene"], "Lysene":["Lyseni"],
                  "Meereen":["Meereenese"], "Northmen":["Northern mountain clans", "northmen"], "Norvoshi": ["Norvos"], "Qartheen": ["Qarth"], "Reach": ["Reachmen"], "Riverlands": ["Rivermen"], "Stormlander":["Stormlands"],
                  "Summer_Islander":["Summer Islands", "Summer Isles"], "Vale": ["Vale mountain clans", "Valemen"], "Westerman":["Westerlands", "westermen", "Westermen"]}

    for key in cult_dict:
        got_raw=got_raw.replace(cult_dict[key], key)

    got_raw["culture"].value_counts()

    #############################################################################
    ##  Variable: house
    # Looking a total missing values in "house column"
    print(got_raw['house'].isnull().sum())

    # Completing some missing values based on character's last name
    name_list = ['Stark', 'Targaryen', 'Durrandon', 'Mudd', 'Tyrell', 'Greyjoy', 'Martell', 'Hoare', 'Frey', 'Lannister', 'Baratheon', 'Florent']
    for house in name_list:
        mask = got_raw['name'].str.contains(house)
        got_raw.loc[mask, 'house'] = str("House " + house)

    # dropping "name" from the dataset
    got_raw.drop(["name"], axis = 1, inplace = True)

    # Looking a total missing values in "house column"
    print(got_raw['house'].isnull().sum())

    # creating category unknown "U" for all characters with missing values in house
    got_raw["house"].fillna('House_U', inplace = True)

    # Grouping houses into 9 main groups

    house_dict= {"House_Crownlands":["House Baratheon", "House Baratheon of King's Landing", "House Baratheon of Dragonstone",
                               "House Blount", "House Byrch", "House Bywater", "House Chelsted", "House Chyttering", "House Farring",
                               "House Gaunt", "House Hayford", "House Mallery", "House Massey", "House Rambton", "House Rosby", "House Rykker",
                               "House Staunton", "House Stokeworth", "House Thorne", "House Brune of Brownhollow", "House Hogg", "House Bar Emmon",
                               "House Celtigar", "House Sunglass", "House Velaryon", "House Velaryon", "House Brune of the Dyre Den", "House Crabb",
                               "House Hardy", "House Hollard", "House Longwaters", "House Blackfyre", "House Darklyn"],
                 "House_North": ["House Stark", "House Ashwood", "House Bole", "House Bolton", "House Branch", "House Glover",
                               "House Cassel", "House Cerwyn", "House Condon", "House Dustin", "House Flint of Flint's Finger",
                               "House Flint of Widow's Watch", "House Forrester", "House Glenmore", "House Holt",
                               "House Hornwood", "House Ironsmith", "House Tallhart", "House Karstark", "House Lake", "House Lightfoot",
                               "House Locke", "House Long", "House Manderly", "House Marsh", "House Mollen","House Mormont",
                               "House Moss", "House Overton", "House Poole", "House Reed", "House Ryswell", "House Slate", "House Stout",
                               "House Umber", "House Waterman", "House Wells", "House Whitehill", "House Woods", "House Woolfield",
                               "House Glover Tallhart", "House Blackmyre", "House Boggs", "House Cray", "House Fenn", "House Greengood",
                               "House Peat","House Quagg", "House Reed", "House Burley", "House Flint of the mountains", "House Harclay",
                               "House Knott", "House Liddle", "House Norrey", "House Wull", "House Crowl", "House Magnar", "House Stane",
                               "House Blackwood", "House Amber", "House Greenwood", "House Fisher of the Stony Shore", "House Flint of Breakstone Hill",
                               "House Frost", "House Greystark", "House Ryder", "House Towers", "House Woodfoot"],
                 "House_Vale": ["House Arryn of the Eyrie", "House Arryn", "House Baelish", "House Belmore", "House Breakstone", "House Coldwater", "House Corbray",
                               "House Crayne", "House Donniger","House Dutton", "House Egen", "House Elesham", "House Grafton", "House Hardyng", "House Hersy",
                               "House Hunter", "House Royce", "House Lipps", "House Lynderly", "House Melcolm", "House Moore", "House Pryor","House Redfort",
                               "House Royce of Runestone", "House Ruthermont","House Tollett","House Upcliff", "House Waynwood",
                               "House Wydman", "House Shett of Gull Tower", "House Shett of Gulltown", "House Templeton","House Waxley",
                               "House Woodhull", "House Borrell", "House Longthorpe", "House Sunderland", "House Torrent", "House Arryn of Gulltown",
                               "House Royce of the Gates of the Moon", "House Extinct Houses", "House Shell", "House Brightstone"],
                 "House_Riverlands": ["House Tully", "House Bigglestone", "House Blackwood", "House Blanetree", "House Bracken", "House Butterwell",
                               "House Chambers", "House Charlton", "House Darry", "House Deddings", "House Erenford", "House Frey", "House Goodbrook",
                               "House Grell", "House Harlton", "House Hawick", "House Keath", "House Lolliston", "House Lychester",
                               "House Mallister", "House Mooton", "House Nutt", "House Pemford", "House Perryn", "House Piper",
                               "House Roote", "House Ryger", "House Shawney", "House Smallwood", "House Terrick", "House Vance of Atranta", "House Vance of Wayfarer's Rest",
                               "House Vypren", "House Wayn", "House Whent", "House Cox", "House Grey", "House Haigh", "House Nayland",
                               "House Paege", "House Wode", "House Heddle", "House Lothston", "House Mudd", "House Strong", "House Fisher",
                               "House Harroway", "House Hoare", "House Hook", "House Justman", "House Qoherys", "House Teague", "House Towers of Harrenhal" ],
                 "House_Westerlands": ["House Lannister of Casterly Rock", "House Algood", "House Banefort", "House Bettley", "House Brax",
                               "House Broom", "House Crakehall", "House Doggett", "House Drox", "House Estren", "House Falwell",
                               "House Farman", "House Ferren", "House Foote", "House Garner", "House Hamell", "House Hawthorne Jast Kenning of Kayce Kyndall",
                               "House Lannister of Lannisport", "House Lefford", "House Lydden", "House Marbrand", "House Moreland",
                               "House Myatt", "House Payne", "House Peckledon", "House Plumm", "House Prester", "House Sarsfield",
                               "House Sarwyck", "House Serrett", "House Stackspear", "House Turnberry", "House Westerling",
                               "House Yarwyck", "House Clegane", "House Clifton", "House Greenfield", "House Hetherspoon",
                               "House Lorch", "House Ruttiger", "House Swyft", "House Vikary", "House Westford", "House Yew",
                               "House Lannett", "House Lantell", "House Lannister", "House Lanny", "House Spicer", "House Casterly",
                               "House Parren", "House Reyne", "House Tarbeck" ],
                 "House_Iron_Islands": ["House Greyjoy", "House Blacktyde", "House Botley", "House Drumm", "House Harlaw",
                               "House Farwynd of Sealskin Point", "House Goodbrother of Hammerhorn", "House Harlaw of Harlaw",
                               "House Kenning of Harlaw", "House Merlyn", "House Myre", "House Orkwood", "House Saltcliffe",
                               "House Sparr", "House Stonehouse", "House Stonetree", "House Sunderly", "House Tawney",
                               "House Volmark", "House Wynch", "House Codd", "House Farwynd of the Lonely Light",
                               "House Goodbrother of Corpse", "House Goodbrother", "House Lake",  "House Goodbrother of Crow",
                               "House Spike", "House Keep", "Goodbrother of Downdelving", "Goodbrother of Orkmont",
                               "Goodbrother of Shatterstone", "House Harlaw of Grey Garden", "House Harlaw of Harridan Hill",
                               "House Harlaw of Harlaw Hall", "House Harlaw of the Tower of Glimmering", "House Humble",
                               "House Ironmaker", "House Netley", "House Sharp", "House Shepherd", "House Weaver",
                               "House Greyiron", "House Hoare"],
                 "House_Reach": ["House Tyrell", "House Ambrose", "House Appleton", "House Ashford", "House Beesbury", "House Osgrey",
                               "House Blackbar", "House Bulwer", "House Caswell", "House Cockshaw", "House Cordwayner", "House Costayne",
                               "House Crane", "House Cuy", "House Florent", "House Fossoway of Cider Hall", "House Hightower",
                               "House Meadows", "House Merryweather", "House Mullendore", "House Oakheart",  "House Peake",
                               "House Redwyne", "House Rowan", "House Shermer", "House Tarly", "House Varner", "House Vyrwel",
                               "House Chester", "House Grimm", "House Hewett", "House Serry", "House Ball", "House Bridges",
                               "House Bushy", "House Cobb", "House Conklyn", "House Dunn", "House Durwell", "House Footly",
                               "House Graceford", "House Graves", "House Hastwyck", "House Hunt", "House Hutcheson",
                               "House Inchfield", "House Kidwell", "House Leygood", "House Lowther", "House Lyberr", "House Middlebury",
                               "House Norcross", "House Norridge", "House Oldflowers", "House Osgrey", "House Orme", "House Pommingham",
                               "House Redding", "House Rhysling", "House Risley", "House Roxton", "House Sloane", "House Stackhouse",
                               "House Uffering", "House Webber", "House Westbrook", "House Willum", "House Woodwright",
                               "House Wythers", "House Yelshire", "House Fossoway of New Barrel", "House Osgrey of Standfast",
                               "House Manderly", "House Gardener", "House Osgrey of Leafy Lake"],
                 "House_Dorne": ["House Martell", "House Allyrion", "House Blackmont", "House Dayne of Starfall", "House Fowler",
                               "House Gargalen", "House Dayne", "House Jordayne", "House Ladybright", "House Manwoody", "House Qorgyle",
                               "House Toland", "House Uller", "House Vaith", "House Wells", "House Wyl", "House Yronwood",
                               "House Dalt", "House Drinkwater", "House Dayne of High Hermitage", "House Santagar", "House Briar",
                               "House Brook", "House Brownhill", "House Dryland", "House Holt", "House Hull", "House Lake",
                               "House Shell", "House Wade"],
                 "House_Stormlands": ["House Baratheon", "House Caron", "House Dondarrion", "House Selmy", "House Swann",
                               "House Noble Houses", "House Buckler", "House Cafferen", "House Errol", "House Estermont",
                               "House Fell", "House Gower", "House Grandison", "House Hasty", "House Herston", "House Horpe",
                               "House Kellington", "House Lonmouth", "House Mertyns", "House Morrigen", "House Musgood",
                               "House Peasebury", "House Penrose", "House Rogers", "House Staedmon", "House Swygert",
                               "House Tarth", "House Trant", "House Tudbury", "House Wagstaff", "House Wensington", "House Whitehead",
                               "House Wylde", "House Bolling", "House Brownhill", "House Connington", "House Hasty",
                               "House Seaworth", "House Cole", "House Durrandon", "House Toyne"],
                 "Nights_Watch": ["Night's Watch" ]
                 }
    for key in house_dict:
        got_raw = got_raw.replace(house_dict[key], key)


    got_raw["house"].value_counts()



    #############################################################################
    ## Variable: title
    # grouping title into 7 groups

    # Queen and princess
    got_raw = got_raw.replace(["Princess",
                               "PrincessQueen",
                               "PrincessQueenDowager Queen",
                               "PrincessSepta",
                               "Queen",
                               "QueenBlack Bride",
                               "QueenDowager Queen"], "Queen_Princess")


    # King and Prince
    got_raw = got_raw.replace(["King in the North",
                               "King of the Iron Islands",
                               "King-Beyond-the-Wall",
                               "King of Winter",
                               "King of Astapor",
                               "King of the Andals",
                               "King",
                               "Prince",
                               "Prince of Dorne",
                               "Prince of Dragonstone",
                               "Prince of the Narrow Sea",
                               "Prince of Winterfell",
                               "Prince of WinterfellHeir to Winterfell"], "King_Prince")


    # Lady
    got_raw = got_raw.replace(["Lady",
                               "Lady Marya",
                               "Lady of Bear Island",
                               "Lady of Darry",
                               "Lady of the Leaves",
                               "Lady of the Vale",
                               "Lady of Torrhen's Square",
                               "LadyQueen",
                               "LadyQueenDowager Queen"], "Lady")


    # Lord
    got_raw = got_raw.replace(["Lord Captain of the Iron Fleet",
                               "Lord Commander of the Night's Watch",
                               "Lord of Atranta",
                               "Lord of Blackhaven",
                               "Lord of Coldmoat",
                               "Lord of Crows Nest",
                               "Lord of Darry",
                               "Lord of Dragonstone",
                               "Lord of Flint's Finger",
                               "Lord of Greyshield",
                               "Lord of Griffin's Roost",
                               "Lord of Hammerhorn",
                               "Lord of Harrenhal",
                               "Lord of Hellholt",
                               "Lord of Honeyholt",
                               "Lord of Iron Holt",
                               "Lord of Kingsgrave",
                               "Lord of Oakenshield",
                               "Lord of Oldcastle",
                               "Lord of Pebbleton",
                               "Lord of Southshield",
                               "Lord of Starfall",
                               "Lord of Sunflower Hall",
                               "Lord of the Crossing",
                               "Lord of the Deep Den",
                               "Lord of the Hornwood",
                               "Lord of the Iron Islands",
                               "Lord of the Marches",
                               "Lord of the Red Dunes",
                               "Lord of the Seven Kingdoms",
                               "Lord of the Snakewood",
                               "Lord of the Ten TowersLord Harlaw of HarlawHarlaw of Harlaw",
                               "Lord of the Tides",
                               "Lord of the Tor",
                               "Lord of White Harbor",
                               "Lord Paramount of the Mander",
                               "Lord Paramount of the Stormlands",
                               "Lord Paramount of the Trident",
                               "Lord Reaper of Pyke",
                               "Lord Seneschal",
                               "Lord Steward",
                               "Lord Steward of the Iron Islands",
                               "Lordsport",
                               "LordWisdom",
                               "Lord",
                               "Khal"], "Lord")


    # Septas
    got_raw = got_raw.replace(["Septon",
                               "Septa"], "Septas")


    # Maester
    got_raw = got_raw.replace(["Good Master",
                               "Master of coin",
                               "Master of Deepwood Motte",
                               "Master of Harlaw Hall",
                               "master of ships",
                               "Master of whisperers",
                               "Master-at-Arms",
                               "Oarmaster",
                               "Archmaester",
                               "Grand Maester"], "Maester")


    # Knight
    got_raw = got_raw.replace(["Knight of Griffin's Roost",
                               "Serthe Knight of Saltpans",
                               "Ser",
                               "SerCastellan of Casterly Rock"], "Knight")

    # Creating unknown title group "Title_U"

    got_raw["title"].fillna('Title_U', inplace = True)


    got_raw["title"].value_counts()


    #############################################################################
    # Variable books
    # Adding a new column to sum the number of books each character appeared in

    total_books = []

    for index, row in got_raw.iterrows():
        total_books.append(row.book1_A_Game_Of_Thrones +
                               row.book2_A_Clash_Of_Kings +
                               row.book3_A_Storm_Of_Swords +
                               row.book4_A_Feast_For_Crows +
                               row.book5_A_Dance_with_Dragons)

    got_raw['total_books'] = total_books
    #############################################################################
    # Creating Dummies for CULTURE & selecting only those with more than 20 observations
    #############################################################################
    got_raw["culture"].value_counts()

    culture_dummies = pd.get_dummies(list(got_raw['culture']), drop_first=True)
    cu = culture_dummies.sum()

    culture_selected_dummies = culture_dummies.loc[:, ['Cult_U',
                                                       'Northmen',
                                                       'Ironborn',
                                                       'Free_Folk',
                                                       'Braavos',
                                                       'Valyrian',
                                                       'Dornish',
                                                       'Vale',
                                                       "Ghiscari",
                                                       "Dothraki",
                                                       "Riverlands"]]

    #############################################################################
    # Dummy treatment
    #############################################################################

    # Creating Dummies for TITLE & selecting only those with more than 20 observations
    got_raw["title"].value_counts()

    title_dummies = pd.get_dummies(list(got_raw['title']), drop_first=True)
    td = title_dummies.sum()

    title_selected_dummies = title_dummies.loc[:, ['Title_U',
                                                   'Knight',
                                                   'Lord',
                                                   'Maester',
                                                   'King_Prince',
                                                   'Septas',
                                                   'Queen_Princess',
                                                   'Lady']]

    # Creating Dummies for HOUSE & selecting only those with more than 20 observations

    got_raw["house"].value_counts()

    house_dummies = pd.get_dummies(list(got_raw['house']), drop_first=True)
    ho = house_dummies.sum()

    house_selected_dummies = house_dummies.loc[:, ['House_U',
                                                   'House Targaryen',
                                                   'House_Riverlands',
                                                   'House_Reach',
                                                   'House_North',
                                                   'House_Crownlands',
                                                   'House_Westerlands',
                                                   "Nights_Watch",
                                                   "House_Iron_Islands",
                                                   "House_Vale",
                                                   "House_Stormlands",
                                                   "House_Dorne"]]

    # Appending selected dummies to original got dataframe

    got_dummies = pd.concat(
        [got_raw.loc[:, :],
         title_selected_dummies, culture_selected_dummies, house_selected_dummies],
        axis=1)

    # cleaning columns with Nan (the original variables for which dummies were created)

    got_clean = got_dummies.drop(columns=['title',
                                          'culture',
                                          'house'
                                          ])

    return got_clean


