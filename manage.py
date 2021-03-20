from flask_script import Manager
from api import create_app

# sets up the app
app = create_app()

manager = Manager(app)

# adds the python manage.py db init, db migrate, db upgrade commands


@manager.command
def runserver():
    app.run(debug=True, host="0.0.0.0", port=5000)


@manager.command
def runworker():
    app.run(debug=False)


if __name__ == "__main__":
    try:
        manager.run()
    except Exception as e:
        raise e
