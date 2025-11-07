# MongoDB Setup for Salefney Shokran API

This project has been migrated from Firebase Firestore to MongoDB. Here's how to set up MongoDB for development and production.

## Local Development Setup

### Option 1: MongoDB Community Server (Local Installation)

1. **Download and Install MongoDB Community Server**
   - Visit: https://www.mongodb.com/try/download/community
   - Download for your operating system
   - Follow the installation wizard

2. **Start MongoDB Service**
   ```bash
   # Windows (if installed as service)
   net start MongoDB
   
   # Windows (manual start)
   mongod --dbpath "C:\data\db"
   
   # macOS (with Homebrew)
   brew services start mongodb-community
   
   # Linux (systemd)
   sudo systemctl start mongod
   ```

3. **Verify MongoDB is Running**
   ```bash
   # Connect to MongoDB shell
   mongosh
   # or older versions
   mongo
   ```

### Option 2: MongoDB Docker Container

1. **Pull and Run MongoDB Docker Container**
   ```bash
   # Pull MongoDB image
   docker pull mongo:latest
   
   # Run MongoDB container
   docker run -d --name mongodb -p 27017:27017 mongo:latest
   ```

2. **Connect to MongoDB in Container**
   ```bash
   docker exec -it mongodb mongosh
   ```

## Environment Variables

Create a `.env` file in your project root with the following variables:

```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017/
MONGODB_DB_NAME=salefney_shokran

# Flask Configuration
FLASK_SECRET_KEY=your-super-secure-secret-key-here

# Optional: Logging Level
LOG_LEVEL=INFO
```

## Production Setup (MongoDB Atlas)

For production, we recommend using MongoDB Atlas (cloud-hosted MongoDB):

1. **Create MongoDB Atlas Account**
   - Visit: https://www.mongodb.com/atlas
   - Create a free account

2. **Create a Cluster**
   - Follow the setup wizard
   - Choose your preferred cloud provider and region

3. **Create Database User**
   - Go to Database Access
   - Create a new user with appropriate permissions

4. **Configure Network Access**
   - Go to Network Access
   - Add your application's IP addresses

5. **Get Connection String**
   - Go to your cluster and click "Connect"
   - Choose "Connect your application"
   - Copy the connection string

6. **Set Environment Variables**
   ```env
   MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/salefney_shokran?retryWrites=true&w=majority
   MONGODB_DB_NAME=salefney_shokran
   FLASK_SECRET_KEY=your-production-secret-key
   ```

## Database Collections

The application uses the following MongoDB collections:

### Users Collection
```json
{
  "_id": ObjectId,
  "email": "user@example.com",
  "username": "username",
  "password": "hashed_password",
  "created_at": ISODate,
  "email_verified": false
}
```

### Predictions Collection
```json
{
  "_id": ObjectId,
  "model": "xgb_model",
  "score": 0.7532,
  "prediction": "High Risk",
  "inputs": {
    "person_age": 30,
    "person_income": 50000,
    // ... other input fields
  },
  "categorical": {
    "Grade": "B",
    "Residence": "RENT",
    // ... other categorical fields
  },
  "timestamp": ISODate,
  "created_at": "2025-11-07T12:00:00Z",
  "user_id": "user_object_id_string" // Optional: only if user is authenticated
}
```

## Running the Application

1. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set Environment Variables**
   - Create `.env` file or set system environment variables

3. **Start the Application**
   ```bash
   python api/index.py
   ```

4. **Test the API**
   - Visit: http://localhost:5000/docs for Swagger documentation
   - Test registration: POST /register
   - Test login: POST /login
   - Test prediction: POST /predict

## Migration Notes

### Changes from Firebase

1. **Authentication**: 
   - Removed Firebase Auth
   - Added password hashing with Werkzeug
   - JWT tokens now include `user_id` instead of `uid`

2. **Database Operations**:
   - Replaced Firestore queries with MongoDB queries
   - Added ObjectId handling for document IDs
   - Updated error handling for MongoDB-specific exceptions

3. **Dependencies**:
   - Removed Firebase-related packages
   - Added pymongo and flask-pymongo

### Breaking Changes

- JWT tokens now use `user_id` field instead of `uid`
- API responses now include `user_id` and `id` fields instead of `uid`
- User registration now requires password (stored as hash)
- Login endpoint now verifies password hash

## Troubleshooting

### Common Issues

1. **Connection Refused**
   - Ensure MongoDB is running on localhost:27017
   - Check if MongoDB service is started

2. **Authentication Failed**
   - Verify MONGODB_URI includes correct username/password
   - Check network access settings in MongoDB Atlas

3. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Activate virtual environment if using one

4. **JWT Token Issues**
   - Ensure FLASK_SECRET_KEY is set and consistent
   - Check token expiration (default: 1 day)

### MongoDB Commands

```bash
# Show databases
show dbs

# Use specific database
use salefney_shokran

# Show collections
show collections

# Query users
db.users.find()

# Query predictions
db.predictions.find()

# Create index on email (for faster queries)
db.users.createIndex({"email": 1}, {"unique": true})

# Create index on user_id for predictions
db.predictions.createIndex({"user_id": 1})
```

## Security Considerations

1. **Always use environment variables for sensitive data**
2. **Use strong, unique secret keys in production**
3. **Enable authentication on MongoDB in production**
4. **Use SSL/TLS for MongoDB connections in production**
5. **Regularly update dependencies for security patches**