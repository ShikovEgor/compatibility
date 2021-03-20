from flask import Blueprint, request
from api.core import create_response, serialize_list, logger
from sqlalchemy import inspect
import json


main = Blueprint("main", __name__)  # initialize blueprint

def init()
# function that is called when you visit /persons
@main.route("/api/v1/services/korchevatel/<int:user_id>", methods=["GET"])
def call_corchevatel(user_id):
    from korchevatel import inferring
	
    res = inferring('users_articles_data_2.json', 'Гулева Вал', 2)


    return create_response(data={'result':res})
