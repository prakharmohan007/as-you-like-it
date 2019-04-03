Parameters:
1. num_user: (Optional) number of users to extract. if greater than the number of users available,
             it will return the number of user available. Default = 2000
2. num_item: (Optional) number of items to extract. If greater than the number of items available,
             it will return the umber of items available. default = 2000
3. dir:      (Optional)target directory. Default = "./data". If path is not provided and directory
             does not exist it will create the directory "./data". If direcory create fails,
             it will save the csv file in currect directory

Que: How to use the script?
Ans: 
To consider default values:
$ python ./make_data_csv.py

To provide custom values
$ python ./make_data_csv.py --num_user <value> --num_item <value> --dir <"./value">
