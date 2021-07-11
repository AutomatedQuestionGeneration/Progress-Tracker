# EDA on QUAC

This has three notebooks as of now covering topic modelling using BERT and another on showing how to use the data.

I have added a `boilerplate.py` which can be used to easily load the data and make the code resuable.
Just upload the `boilerplate.py` using drag and drop in Colab which loads the file in a second.

You can use `get_data("squad")` to get your squad2.0 dataset
Or use `get_data("quac")` to get the quac dataset.
While running this function, you might get an error in the first run, please try to run the cell again and the error should be removed.


`data_to_dataframe(dataset)` converts the data directly to a pandas dataframe.
If you get an error while running this, run this code cell again and the error should be gone.

I haven't added comments right now, so let me know if you use the same boilerplate, then I can add comments.

There are a few more functions I will add later on for lemmatization, stemming etc which has been done in the bertopic modelling notebook so that the code can be reused.


