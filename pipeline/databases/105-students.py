#!/usr/bin/env python3
"""Returns all students sorted by average score"""


from pymongo import MongoClient

def top_students(mongo_collection):
    """Aggregate to calculate the average score for each student and sort them"""
    pipeline = [
        {
            "$unwind": "$scores"
        },
        {
            "$group": {
                "_id": "$_id",
                "name": {"$first": "$name"}, 
                "averageScore": {"$avg": "$scores.score"}
            }
        },
        {
            "$sort": {"averageScore": -1}
        }
    ]
    
    result = mongo_collection.aggregate(pipeline)
    

    students_sorted_by_score = list(result)
    
    return students_sorted_by_score
