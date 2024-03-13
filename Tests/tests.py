# (machines,equipments,duration)

tests = {
    "TC_001" : {
        "jobs" : {
            "job_1" : [([1],[],1)],
            "job_2" : [([2],[],1)],
            "job_3" : [([3],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 1
    },
    "TC_002" : {
        "jobs" : {
            "job_1" : [([1],[],1)],
            "job_2" : [([1],[],1)],
            "job_3" : [([1],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_003" : {
        "jobs" : {
            "job_1" : [([1,2,3],[],1)],
            "job_2" : [([1,2,3],[],1)],
            "job_3" : [([1,2,3],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 1
    },
    "TC_004" : {
        "jobs" : {
            "job_1" : [([1],[],1),([2],[],1),([3],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_005" : {
        "jobs" : {
            "job_1" : [([1],[],1),([1],[],1),([1],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_006" : {
        "jobs" : {
            "job_1" : [([1,2,3],[],1),([1,2,3],[],1),([1,2,3],[],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_007" : {
        "jobs" : {
            "job_1" : [([1],[1],1)],
            "job_2" : [([2],[1],1)],
            "job_3" : [([3],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_008" : {
        "jobs" : {
            "job_1" : [([1],[1],1)],
            "job_2" : [([1],[1],1)],
            "job_3" : [([1],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_009" : {
        "jobs" : {
            "job_1" : [([1,2,3],[1],1)],
            "job_2" : [([1,2,3],[1],1)],
            "job_3" : [([1,2,3],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_010" : {
        "jobs" : {
            "job_1" : [([1],[1],1),([2],[1],1),([3],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_011" : {
        "jobs" : {
            "job_1" : [([1],[1],1),([1],[1],1),([1],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_012" : {
        "jobs" : {
            "job_1" : [([1,2,3],[1],1),([1,2,3],[1],1),([1,2,3],[1],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_013" : {
        "jobs" : {
            "job_1" : [([1],[1],1)],
            "job_2" : [([2],[2],1)],
            "job_3" : [([3],[3],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 1
    },
    "TC_014" : {
        "jobs" : {
            "job_1" : [([1],[1],1)],
            "job_2" : [([1],[2],1)],
            "job_3" : [([1],[3],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_015" : {
        "jobs" : {
            "job_1" : [([1,2,3,4],[1],1)],
            "job_2" : [([1,2,3,4],[2],1)],
            "job_3" : [([1,2,3,4],[3],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_016" : {
        "jobs" : {
            "job_1" : [([1],[1],1),([2],[2],1),([3],[3],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_017" : {
        "jobs" : {
            "job_1" : [([1],[1],1),([2],[2],1),([3],[3],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_018" : {
        "jobs" : {
            "job_1" : [([1,2,3],[1],1),([1,2,3],[2],1),([1,2,3],[3],1)]
        },
        "machine_downtimes" : {

        },
        "timespan": 3
    },
    "TC_019" : {
        "jobs" : {
            "job_1" : [([1],[],1)],
            "job_2" : [([2],[],1)],
            "job_3" : [([3],[],1)]
        },
        "machine_downtimes" : {
            1 : [1,2],
            2 : [0,2],
            3 : [0,1],
        },
        "timespan": 3
    },
    "TC_020" : {
        "jobs" : {
            "job_1" : [([1,2,3],[],1)],
            "job_2" : [([1,2,3],[],1)],
            "job_3" : [([1,2,3],[],1)]
        },
        "machine_downtimes" : {
            1 : [1,2],
            2 : [0,2],
            3 : [0,1]
        },
        "timespan": 3
    },
    "TC_021" : {
        "jobs" : {
            "job_1" : [([1],[],2)],
        },
        "machine_downtimes" : {
            1 : [1] 
        },
        "timespan": 4
    },
}